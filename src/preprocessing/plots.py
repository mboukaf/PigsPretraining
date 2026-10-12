"""Diagnostic figures for the preprocessing steps."""
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .config import FS, SIGNALS, UNITS

SIGNAL_COLOR = "#2a78d6"
INVALID_COLOR = "#e34948"


def _envelope(x: np.ndarray, n_bins: int):
    k = max(len(x) // n_bins, 1)
    xb = x[: (len(x) // k) * k].reshape(-1, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.arange(len(xb)) * k + k / 2, np.nanmin(xb, axis=1), np.nanmax(xb, axis=1)


def plot_validity(raw: np.ndarray, valid: np.ndarray, pig_id: int,
                  signals: list[str] = SIGNALS, n_bins: int = 3000) -> matplotlib.figure.Figure:
    """Full recording per signal (min-max envelope) with invalid stretches shaded."""
    fig, axes = plt.subplots(len(signals), 1, figsize=(13, 1.35 * len(signals)), sharex=True)
    for ax, k in zip(axes, range(len(signals))):
        t, lo, hi = _envelope(raw[k], n_bins)
        ax.fill_between(t / FS / 60, lo, hi, color=SIGNAL_COLOR, linewidth=0)
        bad = ~valid[k]
        if bad.any():
            ymin, ymax = np.nanpercentile(raw[k], [0.5, 99.5])
            ax.set_ylim(ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))
            d = np.diff(np.concatenate([[0], bad.view(np.int8), [0]]))
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                ax.axvspan(s / FS / 60, e / FS / 60, color=INVALID_COLOR, alpha=0.25, linewidth=0)
        ax.set_ylabel(f"{signals[k]}\n[{UNITS[signals[k]]}]", rotation=0, ha="right", va="center", fontsize=8)
        ax.text(1.002, 0.5, f"{100 * valid[k].mean():.1f}% valid", transform=ax.transAxes,
                fontsize=8, va="center", color="#555555")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e5e5e5", linewidth=0.6)
    axes[-1].set_xlabel("time [min]")
    fig.suptitle(f"Pig{pig_id}: Step 2 validity (red = invalid; y-axis clipped to the 0.5–99.5 % range)", y=1.0)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_beat_overlay(filtered: np.ndarray, boundaries: np.ndarray, grid_src: np.ndarray, pig_id: int,
                      signals: list[str] = SIGNALS, show: tuple = ("P_ao", "Q_ao", "P_fem", "P_lv", "V_lv", "P_vc"),
                      n_beats: int = 60) -> matplotlib.figure.Figure:
    """`n_beats` consecutive P_ao-defined beats from mid-recording, overlaid and aligned at their
    start boundary. A good grid gives tightly superimposed curves."""
    mid = len(boundaries) // 2
    idx = [i for i in range(mid, len(boundaries) - 1) if grid_src[i] == 0][:n_beats]
    max_len = int(np.max(np.diff(boundaries)[idx]))
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, s in zip(axes.flat, show):
        k = signals.index(s)
        for i in idx:
            seg = filtered[k, boundaries[i]:boundaries[i + 1]]
            ax.plot(np.arange(len(seg)) / FS * 1000, seg, color=SIGNAL_COLOR, lw=0.6, alpha=0.35)
        ax.set_title(f"{s} [{UNITS[s]}]", fontsize=9)
        ax.set_xlim(0, max_len / FS * 1000)
        ax.axvline(80, color="#888888", lw=0.8, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e5e5e5", linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("time since beat start [ms]")
    fig.suptitle(f"Pig{pig_id}: {len(idx)} consecutive beats overlaid at their boundary "
                 f"(dashed: aortic foot = start + 80 ms)", y=1.0)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_resampled(X: np.ndarray, avail: np.ndarray, rr: np.ndarray, pig_id: int, cardiac: list[str],
                   n_show: int = 300, seed: int = 0) -> matplotlib.figure.Figure:
    """Random available beats after resampling to L points, with the median beat."""
    rng = np.random.default_rng(seed)
    L = X.shape[2]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, (j, s) in zip(axes.flat, enumerate(cardiac)):
        idx = np.flatnonzero(avail[:, j])
        if len(idx) == 0:
            ax.text(0.5, 0.5, "not available", ha="center", va="center", transform=ax.transAxes, color="#888888")
        else:
            show = rng.choice(idx, min(n_show, len(idx)), replace=False)
            for i in show:
                ax.plot(X[i, j], color=SIGNAL_COLOR, lw=0.5, alpha=0.15)
            ax.plot(np.median(X[idx, j], axis=0), color="#1a1a19", lw=1.6)
        ax.set_title(f"{s} [{UNITS[s]}]  ({100 * avail[:, j].mean():.1f}% of beats)", fontsize=9)
        ax.set_xlim(0, L - 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e5e5e5", linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xlabel(f"point within beat (0 … {L - 1})")
    fig.suptitle(f"Pig{pig_id}: {n_show} random beats resampled to {L} points (black = median beat); "
                 f"beat duration {np.median(rr):.2f} s median, {np.percentile(rr, 5):.2f}–{np.percentile(rr, 95):.2f} s (5–95 %)", y=1.0)
    fig.tight_layout()
    plt.close(fig)
    return fig
