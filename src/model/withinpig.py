"""Within-pig evaluation: the probe is fitted on part of a pig's recording and tested on another
part of the SAME pig (Docs/experiments.md, "Within-pig split"). Encoders are not retrained: the
trained encoder is the one of the fold where this pig is the test pig (it never saw the pig).

Splits (windows sorted in time; windows overlapping the test part are dropped from training):
    blocked   5 contiguous blocks, each used once as test (predictions pooled over blocks)
    chrono    first 60 % → last 40 % (the later experimental phases are not seen when fitting)
Scores on the test windows: Pearson r and R² (absolute: calibration is constant within a pig).

Usage (from src/):
    python -m model.withinpig E7          # uses runs/E7/fold<k>/encoder_best.pt for every pig k
Results → runs/withinpig.json.
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from preprocessing.config import ROOT

from data.folds import make_fold
from data.store import select_recordings
from data.windows import STRIDE, W
from model.benchmark import _impute, features
from model.losses import SSLModel
from model.targets import TARGETS, window_targets

REGIMES = {"full": None, "P_ao_only": ["P_ao"]}
GAP = W // STRIDE            # windows overlapping a given window on one side
N_BLOCKS, CHRONO_FRAC = 5, 0.6
OUT = ROOT / "runs" / "withinpig.json"


def _fit_predict(Ftr, ytr, Fte):
    ok = ~np.isnan(ytr)
    if ok.sum() < 30:
        return np.full(len(Fte), np.nan)
    sc = StandardScaler().fit(Ftr[ok])
    model = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(sc.transform(Ftr[ok]), ytr[ok])
    return model.predict(sc.transform(Fte))


def _scores(y, yhat):
    ok = ~np.isnan(y) & ~np.isnan(yhat)
    if ok.sum() < 20 or np.std(y[ok]) == 0:
        return np.nan, np.nan
    r = np.corrcoef(y[ok], yhat[ok])[0, 1] if np.std(yhat[ok]) > 0 else 0.0
    r2 = 1 - np.sum((y[ok] - yhat[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
    return float(r), float(r2)


def splits(n: int) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    idx = np.arange(n)
    out = {"blocked": []}
    edges = np.linspace(0, n, N_BLOCKS + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        test = idx[a:b]
        train = idx[(idx < a - GAP) | (idx >= b + GAP)]
        out["blocked"].append((train, test))
    c = int(CHRONO_FRAC * n)
    out["chrono"] = [(idx[:c - GAP], idx[c:])]
    return out


def run_pig(ckpt_path, device) -> list[dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    k = args["test_pig"]
    recs = select_recordings(args.get("pigs"), args.get("no_flow_pigs"))
    fold = make_fold(recs, k, args.get("val_pig"), vlv_mode=args.get("vlv_mode", "relative"),
                     qao_mode=args.get("qao_mode", "zero"), per_recording=args.get("per_recording", False))
    torch.manual_seed(args.get("seed", 0))
    untrained = SSLModel().encoder.to(device).eval()
    trained = SSLModel().encoder.to(device).eval()
    trained.load_state_dict(ckpt["encoder"])
    ds = fold.test
    order = np.argsort([s for _, s in ds.index])                     # windows in time order
    Y = window_targets(recs, ds.index).iloc[order].reset_index(drop=True)
    rows = []
    for regime, signals in REGIMES.items():
        F = features(ds, {"trained": trained, "untrained": untrained}, signals or list(recs[k].signals), device)
        reps = {"trained": F["trained"], "untrained": F["untrained"],
                "handcrafted": _impute(F["handcrafted"], F["handcrafted"])[1], "mean_beat": F["mean_beat"]}
        for split, folds in splits(len(order)).items():
            for m, R in reps.items():
                R = R[order]
                for t in TARGETS:
                    y = Y[t].to_numpy()
                    pred = np.full(len(y), np.nan)
                    for tr, te in folds:
                        A, B = R[tr], R[te]
                        if m == "mean_beat":                           # PCA fitted on the training part only
                            pca = PCA(32, random_state=0).fit(A)
                            A, B = pca.transform(A), pca.transform(B)
                        pred[te] = _fit_predict(A, y[tr], B)
                    test_idx = np.concatenate([te for _, te in folds])
                    r, r2 = _scores(y[test_idx], pred[test_idx])
                    rows.append({"pig": k, "regime": regime, "split": split,
                                 "method": "template_pca" if m == "mean_beat" else m,
                                 "target": t, "r": r, "r2": r2})
    return rows


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for ckpt_path in sorted((ROOT / "runs" / a.run).glob("fold*/encoder_best.pt")):
        rows += run_pig(ckpt_path, device)
        print(f"done {ckpt_path.parent.name}", flush=True)
    for r in rows:
        r["run"] = a.run
    prev = json.loads(OUT.read_text()) if OUT.exists() else []
    OUT.write_text(json.dumps([r for r in prev if r["run"] != a.run] + rows))


if __name__ == "__main__":
    main()
