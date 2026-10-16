"""Option A2 — does the representation detect a change of state? (spec §15.2, Docs/experiments.md "A2")

The only intervention whose timing is recoverable from the data is the PEEP recruitment
manoeuvre (PEEP stepped up to ≈ 10–20 cmH₂O, read from airway pressure — never an encoder input).
Label: window during a manoeuvre = PEEP ≥ RM_PEEP (model/targets.py, `peep`).

A logistic regression (class-balanced) is fitted on the training pigs' windows and scored on the
test pig with the ROC AUC: how well the cardiac representation alone reveals the manoeuvre.
Methods and regimes as in model/benchmark.py (full input, pressures, P_ao only).

Usage (from src/):
    python -m model.tracking E7 E7s1 E7s2 E9 E9s1 E9s2 --folds 4 6 7 8
Results → runs/tracking.json.
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from preprocessing.config import ROOT

from data.folds import build_fold
from data.store import select_recordings
from model.benchmark import REGIMES, _impute, features
from model.encoder import config_from_args
from model.losses import SSLModel
from model.targets import window_targets

RM_PEEP = 7.0
OUT = ROOT / "runs" / "tracking.json"


def auc(Ftr, ytr, Fte, yte, max_train=6000, seed=0) -> float:
    ok_tr, ok_te = ~np.isnan(ytr), ~np.isnan(yte)
    ytr_b, yte_b = ytr[ok_tr] >= RM_PEEP, yte[ok_te] >= RM_PEEP
    if ytr_b.sum() < 20 or yte_b.sum() < 10 or (~yte_b).sum() < 10:
        return np.nan
    idx = np.random.default_rng(seed).choice(len(ytr_b), min(max_train, len(ytr_b)), replace=False)
    A = Ftr[ok_tr][idx]
    sc = StandardScaler().fit(A)
    clf = LogisticRegressionCV(Cs=np.logspace(-3, 2, 6), class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(A), ytr_b[idx])
    return float(roc_auc_score(yte_b, clf.decision_function(sc.transform(Fte[ok_te]))))


def run_checkpoint(ckpt_path, device) -> list[dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    recs = select_recordings(args.get("pigs"), args.get("no_flow_pigs"))
    fold = build_fold(recs, args)
    torch.manual_seed(args.get("seed", 0))
    untrained = SSLModel(config_from_args(args)).encoder.to(device).eval()
    trained = SSLModel(config_from_args(args)).encoder.to(device).eval()
    trained.load_state_dict(ckpt["encoder"])
    enc = {"trained": trained, "untrained": untrained}
    ytr = window_targets(recs, fold.train.index)["peep"].to_numpy()
    yte = window_targets(recs, fold.test.index)["peep"].to_numpy()
    rows = []
    for regime, signals in REGIMES.items():
        Ftr, Fte = features(fold.train, enc, signals, device), features(fold.test, enc, signals, device)
        mtr, mte = Ftr["has"], Fte["has"]
        pca = PCA(32, random_state=0).fit(Ftr["mean_beat"][mtr])
        hc_tr, hc_te = _impute(Ftr["handcrafted"], Fte["handcrafted"])
        reps = {"trained": (Ftr["trained"], Fte["trained"]), "untrained": (Ftr["untrained"], Fte["untrained"]),
                "handcrafted": (hc_tr, hc_te),
                "template_pca": (pca.transform(Ftr["mean_beat"]), pca.transform(Fte["mean_beat"]))}
        for m, (A, B) in reps.items():
            rows.append({"run": ckpt_path.parent.parent.name, "fold": args["test_pig"], "seed": args.get("seed", 0),
                         "pigs": "paper" if args.get("pigs") else "all", "regime": regime, "method": m,
                         "auc": auc(A[mtr], ytr[mtr], B[mte], yte[mte])})
    return rows


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prev = json.loads(OUT.read_text()) if OUT.exists() else []
    for run in a.runs:
        for ckpt_path in sorted((ROOT / "runs" / run).glob("fold*/encoder_best.pt")):
            k = int(ckpt_path.parent.name[4:])
            if a.folds and k not in a.folds:
                continue
            rows = run_checkpoint(ckpt_path, device)
            prev = [r for r in prev if not (r["run"] == run and r["fold"] == k)] + rows
            OUT.write_text(json.dumps(prev))
            print(f"{run} fold {k}: AUC\n{pd.DataFrame(rows).pivot(index='regime', columns='method', values='auc').round(2).to_string()}", flush=True)


if __name__ == "__main__":
    main()
