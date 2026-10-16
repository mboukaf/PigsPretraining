"""Option A1 — robustness to missing signals (spec §15.3, Docs/experiments.md "A1").

Scenario: a probe (a downstream model) is fitted once, on training pigs with all signals; at
inference on a new pig only a subset S of the signals is available, and nothing is refitted.

For each representation (trained encoder, untrained encoder, handcrafted features, template PCA):
  transfer   probe fitted on FULL-input training windows, applied to the test pig's S-input
             windows (handcrafted: missing features replaced by the training medians; PCA: missing
             signals' mean beats are 0, the normalized centre)
             handcrafted_imputed: the classical alternative — the missing handcrafted features are
             predicted from the available ones by a ridge regression fitted on the training pigs
             (full data), then the full-input probe is applied
  refit      probe fitted on S-input training windows (upper bound; = model/benchmark.py)
  consistency  for the same test window, Pearson r between the standardized full-input and
             S-input representation vectors (1 = the subset gives the same representation)
Scores: within-pig Pearson r on the test pig, per target (model/targets.py).

Usage (from src/):
    python -m model.robustness E7 E7s1 E7s2 E9 E9s1 E9s2 --folds 4 6 7 8
Results → runs/robustness.json.
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

from preprocessing.config import CARDIAC, ROOT

from data.folds import build_fold
from data.store import select_recordings
from model.benchmark import _impute, features
from model.encoder import config_from_args
from model.losses import SSLModel
from model.monitor import center_by
from model.targets import TARGETS, window_targets

SUBSETS = {
    "P_ao_only": ["P_ao"], "P_fem_only": ["P_fem"], "pressures": ["P_ao", "P_fem", "P_lv", "P_vc"],
    "P_ao+Q_ao": ["P_ao", "Q_ao"], "P_lv+V_lv": ["P_lv", "V_lv"], "P_fem+P_vc": ["P_fem", "P_vc"],
}
OUT = ROOT / "runs" / "robustness.json"


def _fit(F, y, max_train=5000, seed=0):
    ok = ~np.isnan(y)
    if ok.sum() < 50:
        return None
    idx = np.flatnonzero(ok)
    idx = np.random.default_rng(seed).choice(idx, min(max_train, len(idx)), replace=False)
    sc = StandardScaler().fit(F[idx])
    return sc, RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(sc.transform(F[idx]), y[idx])


def _score(model, F, y, groups=None):
    """Within-pig Pearson r (centred per pig when the test set holds several pigs)."""
    ok = ~np.isnan(y)
    if model is None or ok.sum() < 20 or np.std(y[ok]) == 0:
        return np.nan
    yhat = model[1].predict(model[0].transform(F[ok]))
    if np.std(yhat) == 0:
        return 0.0
    g = groups[ok] if groups is not None else None
    a, b = center_by(y[ok], g), center_by(yhat, g)
    return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum() + 1e-12))


def _consistency(full: np.ndarray, sub: np.ndarray, ref: np.ndarray) -> float:
    sc = StandardScaler().fit(ref)
    a, b = sc.transform(full), sc.transform(sub)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12)
    return float(np.median(r))


def impute_regression(full_train: np.ndarray, subset_test: np.ndarray) -> np.ndarray:
    """Predict the missing handcrafted features (NaN columns of subset_test) from the available
    ones, with a ridge regression fitted on the full training features; return the imputed matrix
    in the same layout as `_impute` (values + missing indicators, all indicators 0 here)."""
    tr = np.where(np.isnan(full_train), np.nanmedian(full_train, axis=0), full_train)
    tr = np.nan_to_num(tr)
    avail = ~np.all(np.isnan(subset_test), axis=0)
    out = np.where(np.isnan(subset_test), np.nanmedian(full_train, axis=0), subset_test)
    if avail.any() and (~avail).any():
        sc = StandardScaler().fit(tr[:, avail])
        reg = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(sc.transform(tr[:, avail]), tr[:, ~avail])
        x = np.nan_to_num(np.where(np.isnan(subset_test[:, avail]), np.nanmedian(full_train[:, avail], axis=0),
                                   subset_test[:, avail]))
        out[:, ~avail] = reg.predict(sc.transform(x))
    _, imp = _impute(full_train, out)
    return imp


def representations(F: dict, pca: PCA, hc_ref: np.ndarray) -> dict[str, np.ndarray]:
    _, hc = _impute(hc_ref, F["handcrafted"])
    return {"trained": F["trained"], "untrained": F["untrained"], "handcrafted": hc,
            "template_pca": pca.transform(F["mean_beat"])}


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
    ytr, yte = window_targets(recs, fold.train.index), window_targets(recs, fold.test.index)
    pig_te = np.array([p for p, _ in fold.test.index])

    tr_full = features(fold.train, enc, CARDIAC, device)
    te_full = features(fold.test, enc, CARDIAC, device)
    pca = PCA(32, random_state=0).fit(tr_full["mean_beat"])
    R_tr_full = representations(tr_full, pca, tr_full["handcrafted"])
    R_te_full = representations(te_full, pca, tr_full["handcrafted"])
    R_tr_full["handcrafted_imputed"], R_te_full["handcrafted_imputed"] = R_tr_full["handcrafted"], R_te_full["handcrafted"]
    full_models = {m: {t: _fit(R_tr_full[m], ytr[t].to_numpy()) for t in TARGETS} for m in R_tr_full}

    rows = []
    for subset, signals in SUBSETS.items():
        tr_s = features(fold.train, enc, signals, device)
        te_s = features(fold.test, enc, signals, device)
        mtr, mte = tr_s["has"], te_s["has"]
        if mte.sum() < 20:
            continue
        pca_s = PCA(32, random_state=0).fit(tr_s["mean_beat"][mtr])
        R_te_s = representations(te_s, pca, tr_full["handcrafted"])             # transfer: full-input transforms
        R_te_s["handcrafted_imputed"] = impute_regression(tr_full["handcrafted"], te_s["handcrafted"])
        R_tr_full.setdefault("handcrafted_imputed", R_tr_full["handcrafted"])
        R_tr_refit = representations(tr_s, pca_s, tr_s["handcrafted"])            # refit: subset transforms
        R_te_refit = representations(te_s, pca_s, tr_s["handcrafted"])
        R_tr_refit["handcrafted_imputed"], R_te_refit["handcrafted_imputed"] = R_tr_refit["handcrafted"], R_te_refit["handcrafted"]
        R_te_full.setdefault("handcrafted_imputed", R_te_full["handcrafted"])
        for m in R_te_s:
            cons = _consistency(R_te_full[m][mte], R_te_s[m][mte], R_tr_full[m])
            for t in TARGETS:
                y = yte[t].to_numpy()[mte]
                rows.append({"run": ckpt_path.parent.parent.name, "fold": args["test_pig"], "seed": args.get("seed", 0),
                             "pigs": "paper" if args.get("pigs") else "all", "subset": subset, "method": m,
                             "target": t, "consistency": cons,
                             "r_transfer": _score(full_models[m][t], R_te_s[m][mte], y, pig_te[mte]),
                             "r_refit": _score(_fit(R_tr_refit[m][mtr], ytr[t].to_numpy()[mtr]), R_te_refit[m][mte], y, pig_te[mte]),
                             "r_full": _score(full_models[m][t], R_te_full[m], yte[t].to_numpy(), pig_te)})
    return rows


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
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
            df = pd.DataFrame(rows)
            print(f"{run} fold {k}: median r_transfer\n"
                  f"{df.groupby(['subset', 'method'])['r_transfer'].median().unstack().round(2).to_string()}", flush=True)


if __name__ == "__main__":
    main()
