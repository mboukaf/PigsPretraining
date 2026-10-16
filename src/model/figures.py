"""Result figures (Docs/experiments.md) → figures/results/.

Usage (from src/):
    python -m model.figures
Needs runs/benchmark.json, runs/robustness.json, runs/tracking.json (benchmark / robustness /
tracking scripts) for figures 2–4; figure 1 is recomputed from the data.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from preprocessing.config import CARDIAC, PIGS, ROOT

from data.folds import make_fold
from data.store import select_recordings
from model.benchmark import _impute
from model.features import window_summaries
from model.targets import window_targets

OUT = ROOT / "figures" / "results"
RUNS = ["E9", "E9s1", "E9s2"]          # encoders trained on the project pigs only (4, 6, 7, 8), seeds 0–2
TEST_PIGS = PIGS

# validated categorical palette (light mode), fixed order; text never takes series colours
C = {"trained": "#2a78d6", "handcrafted": "#eb6834", "untrained": "#1baf7a", "template_pca": "#eda100",
     "handcrafted_imputed": "#e87ba4"}
LABEL = {"trained": "Trained encoder", "untrained": "Untrained encoder", "handcrafted": "Handcrafted statistics",
         "template_pca": "Mean-beat PCA", "handcrafted_imputed": "Handcrafted + regression imputation"}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
})


def _load(name: str) -> pd.DataFrame:
    df = pd.DataFrame(json.loads((ROOT / "runs" / f"{name}.json").read_text()))
    return df[df.run.isin(RUNS) & df.fold.isin(TEST_PIGS)]


# --- figure 1: relative cardiac output tracked from aortic pressure only ---------------------

def _summaries(ds, signals):
    keep = torch.tensor([s in signals for s in CARDIAC])
    out = []
    for b in DataLoader(ds, batch_size=512):
        a = b["a"] & keep
        out.append(window_summaries(b["X"] * a.unsqueeze(-1), b["r"], a).numpy())
    return np.concatenate(out)


def fig_flow_tracking():
    recs = select_recordings(PIGS)
    fig, axes = plt.subplots(len(TEST_PIGS), 1, figsize=(9.5, 8.6), sharex=False)
    for ax, k in zip(axes, TEST_PIGS):
        fold = make_fold(recs, k, vlv_mode="scaled", qao_mode="scaled")
        Ftr, Fte = _summaries(fold.train, ["P_ao"]), _summaries(fold.test, ["P_ao"])
        Ftr, Fte = _impute(Ftr, Fte)
        ytr = window_targets(recs, fold.train.index)
        yte = window_targets(recs, fold.test.index)
        ok = ~np.isnan(ytr["co_rel"].to_numpy())
        sc = StandardScaler().fit(Ftr[ok])
        model = RidgeCV(alphas=np.logspace(-3, 4, 15)).fit(sc.transform(Ftr[ok]), ytr["co_rel"].to_numpy()[ok])
        y, yhat = yte["co_rel"].to_numpy(), model.predict(sc.transform(Fte))
        t = np.array([fold.test.data[k].t_beat[s] for _, s in fold.test.index]) / 60
        m = ~np.isnan(y)
        z = lambda v: (v - v[m].mean()) / v[m].std()
        r = np.corrcoef(y[m], yhat[m])[0, 1]

        rm = yte["peep"].to_numpy() >= 7
        for t0 in t[rm]:
            ax.axvspan(t0, t0 + 0.5, color="#d9d8d1", alpha=0.5, lw=0)
        ax.plot(t[m], z(y)[m], color=INK, lw=1.0, label="Actual (aortic flow probe)")
        ax.plot(t[m], z(yhat)[m], color=C["trained"], lw=1.0, label="Predicted from aortic pressure only")
        ax.set_ylabel("z-score")
        ax.set_title(f"Test pig {k}  ·  probe trained on pigs {fold.train_pigs}  ·  within-pig r = {r:.2f}",
                     loc="left", fontsize=9)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("time [min]   (grey bands: PEEP recruitment manoeuvres, PEEP ≥ 7 cmH₂O)")
    fig.suptitle("Relative cardiac output tracked from the aortic pressure waveform alone, in unseen pigs",
                 x=0.01, y=0.995, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.958, "Linear probe on handcrafted beat statistics of P_ao, fitted on two of the other project pigs "
             "(the fold's validation pig is held out). Both curves z-scored within the test pig.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUT / "fig1_flow_tracking.png")
    plt.close(fig)


# --- figure 2: benchmark, encoder vs baselines -------------------------------------------------

def fig_benchmark():
    df = _load("benchmark")
    methods = ["trained", "untrained", "handcrafted", "template_pca"]
    rows = [("All 6 signals", lambda d: d.regime == "full"),
            ("Pressures only", lambda d: d.regime == "pressures"),
            ("P_ao only", lambda d: d.regime == "P_ao_only"),
            ("P_ao only,\ncross-signal targets", lambda d: (d.regime == "P_ao_only") & d.cross_signal)]
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.8))
    axes = [ax]
    for ax, title in zip(axes, ["Leave-one-pig-out over pigs 4, 6, 7, 8"]):
        d = df
        for i, (name, mask) in enumerate(rows):
            for j, m in enumerate(methods):
                vals = d[mask(d) & (d.method == m)].groupby("seed")["r_within"].median()
                y = i + (j - 1.5) * 0.17
                ax.errorbar(vals.mean(), y, xerr=vals.std() if len(vals) > 1 else None, fmt="o", ms=6,
                            color=C[m], ecolor=C[m], elinewidth=1.5, capsize=0,
                            label=LABEL[m] if i == 0 else None, mec="white", mew=1)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0.3, 1.02)
        ax.set_xlabel("within-pig correlation")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("21 physiological targets on unseen pigs:\nsimple beat statistics beat the learned embedding",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.84, "Median over targets and test pigs; dot = mean over 3 seeds, line = ± sd.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 0.83))
    fig.savefig(OUT / "fig2_benchmark.png")
    plt.close(fig)


# --- figure 3: robustness to missing signals ---------------------------------------------------

def fig_robustness():
    d = _load("robustness")
    subsets = ["P_ao_only", "P_fem_only", "P_fem+P_vc", "P_lv+V_lv", "pressures", "P_ao+Q_ao"]
    names = {"P_ao_only": "P_ao", "P_fem_only": "P_fem", "P_fem+P_vc": "P_fem + P_vc",
             "P_lv+V_lv": "P_lv + V_lv", "pressures": "4 pressures", "P_ao+Q_ao": "P_ao + Q_ao"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    methods = ["trained", "handcrafted", "handcrafted_imputed"]
    x = np.arange(len(subsets))
    for j, m in enumerate(methods):
        g = d[d.method == m].groupby(["subset", "seed"])["r_transfer"].median().groupby("subset").agg(["mean", "std"])
        g = g.reindex(subsets)
        axes[0].bar(x + (j - 1) * 0.27, g["mean"], width=0.25, color=C[m], label=LABEL[m],
                    yerr=g["std"].fillna(0), error_kw={"elinewidth": 1, "ecolor": MUTED})
    axes[0].set_xticks(x, [names[s] for s in subsets], rotation=20)
    axes[0].set_ylabel("within-pig correlation")
    axes[0].set_title("Probe fitted with all signals, applied to a subset\n(no refitting)", loc="left")
    axes[0].set_ylim(0, 1)

    cons = d.drop_duplicates(["run", "fold", "subset", "method"])
    for j, m in enumerate(["trained", "handcrafted_imputed", "untrained", "handcrafted"]):
        g = cons[cons.method == m].groupby("subset")["consistency"].median().reindex(subsets)
        axes[1].bar(x + (j - 1.5) * 0.2, g.values, width=0.19, color=C[m], label=LABEL[m])
    axes[1].set_xticks(x, [names[s] for s in subsets], rotation=20)
    axes[1].set_ylabel("r(full-input, subset representation)")
    axes[1].set_title("Consistency of the representation\nacross sensor subsets", loc="left")
    axes[1].set_ylim(-0.1, 1.05)
    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    order = [LABEL[m] for m in ("trained", "handcrafted_imputed", "handcrafted", "untrained")]
    fig.legend([handles[l] for l in order], order, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Missing signals: the encoder is the most sensor-invariant, but a classical imputation pipeline predicts better",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.9, "Leave-one-pig-out over pigs 4, 6, 7, 8; mean ± sd over 3 seeds of the median over targets and pigs.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    fig.savefig(OUT / "fig3_robustness.png")
    plt.close(fig)


# --- figure 4: PEEP-manoeuvre detection --------------------------------------------------------

def fig_tracking():
    df = _load("tracking")
    methods = ["trained", "untrained", "handcrafted", "template_pca"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, regime, title in zip(axes, ["full", "P_ao_only"], ["All 6 signals", "Aortic pressure only"]):
        d = df[df.regime == regime]
        x = np.arange(len(TEST_PIGS))
        for j, m in enumerate(methods):
            g = d[d.method == m].groupby("fold")["auc"].agg(["mean", "std"]).reindex(TEST_PIGS)
            ax.bar(x + (j - 1.5) * 0.2, g["mean"], width=0.19, color=C[m], label=LABEL[m],
                   yerr=g["std"].fillna(0), error_kw={"elinewidth": 1, "ecolor": MUTED})
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(x, [f"test pig {k}" for k in TEST_PIGS])
        ax.set_ylim(0.2, 1.0)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("ROC AUC")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Detecting PEEP recruitment manoeuvres from the cardiac waveforms",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.89, "Leave-one-pig-out over pigs 4, 6, 7, 8; classifier fitted on the training pigs; bar = mean over 3 seeds, "
             "line = ± sd; dashed line = chance.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.87))
    fig.savefig(OUT / "fig4_peep_detection.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_flow_tracking()
    fig_benchmark()
    fig_robustness()
    fig_tracking()
    print(f"figures written to {OUT.relative_to(ROOT)}")
