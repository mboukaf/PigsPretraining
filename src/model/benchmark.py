"""Benchmark of window representations with linear probes (spec §15.1, §15.5).

Methods (all fed the same normalized windows, restricted to the regime's input signals):
    trained      encoder from a run's checkpoint (embedding e, 32-d)
    untrained    same architecture, initialization of that run (epoch 0)
    trained_H    diagnostic: the trained encoder's beat-token outputs (128-d), averaged over
                 attended beats — tests whether information is lost at the 32-d embedding head
    handcrafted+trained  diagnostic: does e add anything to the handcrafted features?
    handcrafted  per kept signal: mean / min / max / peak-to-peak / max slope of the beats,
                 averaged over the window, + beat-to-beat std of the peak-to-peak; mean and std of
                 the beat durations. Missing signals: training median + missing indicator.
    template_pca mean beat of each kept signal over the window (6 × 64), PCA → 32 dims fitted on
                 the training pigs.
Input regimes:
    full        all 6 cardiac signals
    pressures   P_ao, P_fem, P_lv, P_vc (no flow, no volume)
    P_ao_only   aortic pressure only (bedside-like)
Targets: model/targets.py, computed from the FULL data, so a target whose source signal is not
in the regime tests cross-signal information.
Probe: ridge regression (α by internal CV) fitted on the training pigs' windows, scored on the
test pig: within-pig R² and within-pig Pearson r (centred on the test pig; r is scale-free).

Usage (from src/):
    python -m model.benchmark E7 E7s1 E7s2 E9 E9s1 E9s2 --folds 4 6 7 8
Results are appended to runs/benchmark.json (one row per run × fold × method × regime × target).
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
from torch.utils.data import DataLoader

from preprocessing.config import CARDIAC, ROOT

from data.folds import make_fold
from data.store import select_recordings
from model.features import window_summaries
from model.losses import SSLModel
from model.targets import TARGET_SOURCES, TARGETS, window_targets

REGIMES = {
    "full": CARDIAC,
    "pressures": ["P_ao", "P_fem", "P_lv", "P_vc"],
    "P_ao_only": ["P_ao"],
}
OUT = ROOT / "runs" / "benchmark.json"


def _handcrafted(X: torch.Tensor, r: torch.Tensor, a: torch.Tensor) -> np.ndarray:
    """Handcrafted baseline = the window summaries of model/features.py (NaN for absent signals)."""
    return window_summaries(X, r, a).numpy()


def _mean_beat(X: torch.Tensor, a: torch.Tensor) -> np.ndarray:
    w = a.float().unsqueeze(-1)
    return ((X * w).sum(1) / w.sum(1).clamp(min=1)).flatten(1).numpy()     # (B, M·L), 0 if absent


@torch.no_grad()
def features(ds, encoders: dict, signals: list[str], device) -> dict[str, np.ndarray]:
    keep = torch.tensor([s in signals for s in CARDIAC])
    out = {k: [] for k in [*encoders, "trained_H", "handcrafted", "mean_beat", "has"]}
    for b in DataLoader(ds, batch_size=256, shuffle=False):
        a = b["a"] & keep
        has = a.flatten(1).any(1)
        a_enc = torch.where(has[:, None, None], a, b["a"])                  # placeholder; dropped later
        X = b["X"] * a.unsqueeze(-1)
        for name, enc in encoders.items():
            o = enc(X.to(device), b["r"].to(device), a_enc.to(device))
            out[name].append(o["e"].cpu().numpy())
            if name == "trained":
                attended = (~o["pad"][:, 1:]).float().unsqueeze(-1)
                pooled = (o["H"][:, 1:] * attended).sum(1) / attended.sum(1).clamp(min=1)
                out["trained_H"].append(pooled.cpu().numpy())
        out["handcrafted"].append(_handcrafted(X, b["r"], a))
        out["mean_beat"].append(_mean_beat(X, a))
        out["has"].append(has.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def _impute(train: np.ndarray, test: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # all-NaN columns = absent signal
        med = np.nanmedian(train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    keep = ~np.all(np.isnan(train), axis=0)
    f = lambda x: np.concatenate([np.where(np.isnan(x), med, x), np.isnan(x).astype(float)], axis=1)[:, np.r_[keep, keep]]
    return f(train), f(test)


def probe(Ftr, ytr, Fte, yte, max_train: int = 5000, seed: int = 0) -> dict[str, float]:
    tr, te = ~np.isnan(ytr), ~np.isnan(yte)
    if tr.sum() < 50 or te.sum() < 20 or np.std(yte[te]) == 0:
        return {"r2_within": np.nan, "r_within": np.nan}
    idx = np.flatnonzero(tr)
    idx = np.random.default_rng(seed).choice(idx, min(max_train, len(idx)), replace=False)
    sc = StandardScaler().fit(Ftr[idx])
    model = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(sc.transform(Ftr[idx]), ytr[idx])
    y, yhat = yte[te], model.predict(sc.transform(Fte[te]))
    yc, hc = y - y.mean(), yhat - yhat.mean()
    r2 = 1 - np.sum((yc - hc) ** 2) / np.sum(yc ** 2)
    r = np.sum(yc * hc) / np.sqrt(np.sum(yc ** 2) * np.sum(hc ** 2)) if np.std(yhat) > 0 else 0.0
    return {"r2_within": float(r2), "r_within": float(r)}


def benchmark_checkpoint(ckpt_path, device) -> list[dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    recs = select_recordings(args.get("pigs"), args.get("no_flow_pigs"))
    fold = make_fold(recs, args["test_pig"], args.get("val_pig"),
                     vlv_mode=args.get("vlv_mode", "relative"), qao_mode=args.get("qao_mode", "zero"))
    torch.manual_seed(args.get("seed", 0))
    untrained = SSLModel().encoder.to(device).eval()
    trained = SSLModel().encoder.to(device).eval()
    trained.load_state_dict(ckpt["encoder"])
    encoders = {"trained": trained, "untrained": untrained}

    ytr = window_targets(recs, fold.train.index)
    yte = window_targets(recs, fold.test.index)
    rows = []
    for regime, signals in REGIMES.items():
        Ftr, Fte = features(fold.train, encoders, signals, device), features(fold.test, encoders, signals, device)
        mtr, mte = Ftr["has"], Fte["has"]
        pca = PCA(32, random_state=0).fit(Ftr["mean_beat"][mtr])
        hc_tr, hc_te = _impute(Ftr["handcrafted"], Fte["handcrafted"])
        reps = {"trained": (Ftr["trained"], Fte["trained"]), "untrained": (Ftr["untrained"], Fte["untrained"]),
                "trained_H": (Ftr["trained_H"], Fte["trained_H"]),
                "handcrafted": (hc_tr, hc_te),
                "handcrafted+trained": (np.hstack([hc_tr, Ftr["trained"]]), np.hstack([hc_te, Fte["trained"]])),
                "template_pca": (pca.transform(Ftr["mean_beat"]), pca.transform(Fte["mean_beat"]))}
        for method, (A, B) in reps.items():
            for t in TARGETS:
                res = probe(A[mtr], ytr[t].to_numpy()[mtr], B[mte], yte[t].to_numpy()[mte])
                cross = bool(TARGET_SOURCES[t]) and not any(s in signals for s in TARGET_SOURCES[t])
                rows.append({"run": ckpt_path.parent.parent.name, "fold": args["test_pig"],
                             "seed": args.get("seed", 0), "pigs": "paper" if args.get("pigs") else "all",
                             "method": method, "regime": regime, "target": t,
                             "cross_signal": cross} | res)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prev = json.loads(OUT.read_text()) if OUT.exists() else []
    for run in args.runs:
        for ckpt_path in sorted((ROOT / "runs" / run).glob("fold*/encoder_best.pt")):
            k = int(ckpt_path.parent.name[4:])
            if args.folds and k not in args.folds:
                continue
            rows = benchmark_checkpoint(ckpt_path, device)
            prev = [r for r in prev if not (r["run"] == run and r["fold"] == k)] + rows
            OUT.write_text(json.dumps(prev))
            df = pd.DataFrame(rows)
            summ = df.groupby(["regime", "method"])["r_within"].median().unstack().round(2)
            print(f"{run} fold {k}: median within-pig r over targets\n{summ.to_string()}", flush=True)


if __name__ == "__main__":
    main()
