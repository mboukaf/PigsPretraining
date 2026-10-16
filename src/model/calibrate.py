"""Calibrate-then-track: relative cardiac output from aortic pressure only, after calibrating on
the first D minutes of the same pig (Docs/experiments.md, "Calibrate then track").

For each project pig (4, 6, 7, 8) and each calibration duration D:
  calibration  windows starting in [0, D) min (flow known)
  tracking     windows starting after D + one window length (flow only used for scoring)
Representation: handcrafted beat statistics of P_ao (model/features.py) — the best method so far.
Methods:
  population      ridge probe fitted on the OTHER pigs, no calibration
  population+offset  population prediction shifted by its mean error on the calibration period
                  (one-point calibration: the population slope is kept)
  population+cal  population prediction linearly rescaled (y ≈ a·ŷ + b) on the calibration period
  calibrated      ridge probe fitted on the calibration period only
Metrics on the tracking period (target: relative cardiac output `co_rel`, model/targets.py):
  r            Pearson correlation
  r2           absolute R²
  err_pct      RMSE / mean(actual) × 100
  concordance  % of consecutive 5-min changes with the same sign in prediction and actual,
               excluding changes smaller than EXCL_PCT % of the actual value (trend analysis)

Usage (from src/):
    python -m model.calibrate
Results → runs/calibrate.json; figure → figures/results/fig5_calibrate_track.png.
"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, ConcatDataset

from preprocessing.config import CARDIAC, PIGS, ROOT

from data.folds import make_fold
from data.store import select_recordings
from data.windows import W
from model.benchmark import _impute
from model.features import window_summaries
from model.targets import window_targets

DURATIONS_MIN = [5, 10, 20, 30, 45, 60]
TARGET = "co_rel"
BIN_MIN, EXCL_PCT = 5.0, 5.0
OUT = ROOT / "runs" / "calibrate.json"
FIG = ROOT / "figures" / "results" / "fig5_calibrate_track.png"


def p_ao_features(ds) -> np.ndarray:
    keep = torch.tensor([s == "P_ao" for s in CARDIAC])
    out = []
    for b in DataLoader(ds, batch_size=512):
        a = b["a"] & keep
        out.append(window_summaries(b["X"] * a.unsqueeze(-1), b["r"], a).numpy())
    return np.concatenate(out)


def ridge(F, y):
    ok = ~np.isnan(y)
    sc = StandardScaler().fit(F[ok])
    return sc, RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(sc.transform(F[ok]), y[ok])


def predict(model, F):
    return model[1].predict(model[0].transform(F))


def concordance(t_min, y, yhat):
    bins = np.floor(t_min / BIN_MIN)
    df = pd.DataFrame({"b": bins, "y": y, "h": yhat}).dropna().groupby("b").mean()
    dy, dh = np.diff(df["y"].to_numpy()), np.diff(df["h"].to_numpy())
    big = np.abs(dy) >= EXCL_PCT / 100 * np.abs(df["y"].to_numpy()[:-1])
    if big.sum() < 3:
        return np.nan, int(big.sum())
    return float(100 * np.mean(np.sign(dy[big]) == np.sign(dh[big]))), int(big.sum())


def metrics(t_min, y, yhat) -> dict:
    ok = ~np.isnan(y) & ~np.isnan(yhat)
    y, yhat, t_min = y[ok], yhat[ok], t_min[ok]
    conc, n = concordance(t_min, y, yhat)
    return {"r": float(np.corrcoef(y, yhat)[0, 1]),
            "r2": float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)),
            "err_pct": float(100 * np.sqrt(np.mean((y - yhat) ** 2)) / np.mean(y)),
            "concordance": conc, "n_changes": n, "n_windows": int(len(y))}


def run():
    recs = select_recordings()
    rows, examples = [], {}
    for k in PIGS:
        fold = make_fold(recs, k, vlv_mode="scaled", qao_mode="scaled")
        pop_ds = ConcatDataset([fold.train, fold.val])
        pop_index = fold.train.index + fold.val.index
        F_pop, y_pop = p_ao_features(pop_ds), window_targets(recs, pop_index)[TARGET].to_numpy()
        F_k = p_ao_features(fold.test)
        F_pop, F_k = _impute(F_pop, F_k)
        y_k = window_targets(recs, fold.test.index)[TARGET].to_numpy()
        t_k = np.array([fold.test.data[k].t_beat[s] for _, s in fold.test.index]) / 60
        win_min = W * np.median(recs[k].rr) / 60
        pop = ridge(F_pop, y_pop)
        yhat_pop = predict(pop, F_k)
        for D in DURATIONS_MIN:
            cal = (t_k < D) & ~np.isnan(y_k)
            trk = t_k >= D + win_min
            if cal.sum() < 20 or trk.sum() < 50:
                continue
            lr = LinearRegression().fit(yhat_pop[cal, None], y_k[cal])
            preds = {"population": yhat_pop,
                     "population+offset": yhat_pop + np.mean(y_k[cal] - yhat_pop[cal]),
                     "population+cal": lr.predict(yhat_pop[:, None]),
                     "calibrated": predict(ridge(F_k[cal], y_k[cal]), F_k)}
            for m, yh in preds.items():
                rows.append({"pig": k, "D_min": D, "method": m, "n_cal": int(cal.sum())}
                            | metrics(t_k[trk], y_k[trk], yh[trk]))
            if D == 30:
                examples[k] = (t_k, y_k, preds, D)
    OUT.write_text(json.dumps(rows))
    return pd.DataFrame(rows), examples


def figure(df: pd.DataFrame, examples: dict):
    C = {"population": "#1baf7a", "population+offset": "#2a78d6", "population+cal": "#eda100", "calibrated": "#eb6834"}
    LBL = {"population": "Population probe (other pigs, no calibration)",
           "population+offset": "Population probe + offset calibration on this pig",
           "population+cal": "Population probe + slope & offset calibration",
           "calibrated": "Probe fitted on this pig's calibration period only"}
    INK, MUTED = "#1a1a19", "#6b6a63"
    fig = plt.figure(figsize=(11, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.45, top=0.83, bottom=0.07, left=0.07, right=0.98)
    top = gs[0].subgridspec(1, 3, wspace=0.32)
    bottom = gs[1].subgridspec(1, 2, wspace=0.2)
    for i, (col, ylab, ylim) in enumerate([("r", "Pearson r (tracking period)", (-0.2, 1)),
                                           ("err_pct", "RMSE, % of mean CO", (0, None)),
                                           ("concordance", "trend concordance, %", (40, 100))]):
        ax = fig.add_subplot(top[0, i])
        for m in C:
            g = df[df.method == m].groupby("D_min")[col]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.plot(med.index, med.values, "-o", color=C[m], ms=4, lw=1.6, label=LBL[m])
            ax.fill_between(med.index, q1.values, q3.values, color=C[m], alpha=0.12, lw=0)
        ax.set_xlabel("calibration duration [min]")
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e6e5df", lw=0.6)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.925))

    ex = [k for k in (6, 4) if k in examples][:2]
    for j, k in enumerate(ex):
        ax = fig.add_subplot(bottom[0, j])
        t, y, preds, D = examples[k]
        m = ~np.isnan(y)
        ax.axvspan(0, D, color="#d9d8d1", alpha=0.5, lw=0)
        ax.plot(t[m], y[m], color=INK, lw=1, label="actual")
        for meth in ("population", "population+offset"):
            ax.plot(t[m], preds[meth][m], color=C[meth], lw=1)
        ax.set_title(f"Pig {k}: first {D} min used for calibration (grey)", loc="left", fontsize=9)
        ax.set_xlabel("time [min]")
        ax.set_ylabel("relative cardiac output")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e6e5df", lw=0.6)
    fig.suptitle("Calibrate, then track: relative cardiac output from the aortic pressure waveform",
                 x=0.01, y=0.995, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.955, "Handcrafted P_ao beat statistics, ridge probe. Top: tracking period after the calibration "
             "window; line = median, band = interquartile range over pigs 4, 6, 7, 8 (offset calibration does not change r or\n"
             "concordance, so its line covers the population one). Bottom: black = actual, green = population, "
             "blue = population + offset calibration.", fontsize=8, color=MUTED, va="top")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    df, examples = run()
    figure(df, examples)
    pd.set_option("display.width", 220)
    for col in ("r", "r2", "err_pct", "concordance"):
        print(f"\n== {col}: median over pigs {PIGS}\n",
              df.groupby(["method", "D_min"])[col].median().unstack().round(2).to_string())
    print("\nmedian number of 5-min changes scored for concordance:", df["n_changes"].median())
    print(f"figure: {FIG.relative_to(ROOT)}")
