"""Step 1: data inventory and sanity checks (spec §3).

One row per pig: duration, missing data, heart rate, mean pressure, stroke volume,
upstroke timing between channels, and flags for known problems.

The beat detection used here is a rough one for sanity checks only; the real beat
grid is Step 4.

Usage:
    python -m preprocessing.inventory          (from src/)
"""
import numpy as np
import pandas as pd
from scipy import signal

from .config import PIGS, CARDIAC, FS, PAPER_ID, PROC_DIR, SIGNALS
from .io import load_raw

LAG_SEARCH_S = 0.30  # search window around each P_ao upstroke for other channels' upstrokes


def _lowpass(x: np.ndarray, cutoff: float = 25.0) -> np.ndarray:
    b, a = signal.butter(4, cutoff, fs=FS)
    filled = np.where(np.isnan(x), np.nanmedian(x), x)
    return signal.filtfilt(b, a, filled)


def rough_beats(p_ao: np.ndarray) -> np.ndarray:
    """Sample indices of the max upstroke slope of each beat of (filtered) P_ao."""
    dp = np.gradient(p_ao)
    peaks, _ = signal.find_peaks(dp, distance=int(0.45 * FS), height=np.percentile(dp, 99) * 0.5)
    return peaks


def upstroke_lags_ms(x: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Per beat: time of max slope of `x` relative to the P_ao upstroke, in ms."""
    dx = np.gradient(x)
    h = int(LAG_SEARCH_S * FS)
    beats = beats[(beats > h) & (beats < len(x) - h)]
    windows = dx[beats[:, None] + np.arange(-h, h)]
    return (windows.argmax(axis=1) - h) / FS * 1000


def inventory_row(pig_id: int) -> dict:
    raw = load_raw(pig_id)
    sig = dict(zip(SIGNALS, raw))
    filt = {k: _lowpass(sig[k]) for k in CARDIAC}

    beats = rough_beats(filt["P_ao"])
    rr = np.diff(beats) / FS
    q_ml_s = filt["Q_ao"] * 1000 / 60                                # L/min -> mL/s
    sv = np.add.reduceat(q_ml_s, beats)[:-1] / FS                    # mL per beat

    row = {
        "pig": pig_id,
        "paper_id": PAPER_ID.get(pig_id),
        "n_samples": raw.shape[1],
        "duration_min": round(raw.shape[1] / FS / 60, 1),
        "n_beats_rough": len(beats),
        "hr_bpm": round(60 / np.median(rr), 1),
        "mean_P_ao": round(float(np.nanmean(sig["P_ao"])), 1),
        "mean_Q_ao_L_min": round(float(np.nanmean(sig["Q_ao"])), 2),
        "sv_mL": round(float(np.median(sv)), 1),
        "mean_P_vc": round(float(np.nanmean(sig["P_vc"])), 1),
    }
    for k in ["P_lv", "Q_ao", "P_fem"]:
        lags = upstroke_lags_ms(filt[k], beats)
        q25, q50, q75 = np.percentile(lags, [25, 50, 75])
        row[f"lag_{k}_ms"] = round(q50)
        row[f"lag_{k}_iqr_ms"] = round(q75 - q25)
    for k in SIGNALS:
        row[f"nan_%_{k}"] = round(100 * float(np.isnan(sig[k]).mean()), 3)

    # known problems
    flags = []
    if np.nanmin(sig["Q_ao"]) >= 0:
        flags.append("Q_ao never negative (rectified?)")
    if np.nanmin(sig["Q_aw"]) >= 0:
        flags.append("Q_aw never negative (expiration missing)")
    for k in ["P_lv", "Q_ao", "P_fem"]:
        if row[f"lag_{k}_iqr_ms"] > 50:
            flags.append(f"{k} timing unstable")
    row["flags"] = "; ".join(flags)
    return row


def build_inventory(pig_ids: list[int] = PIGS) -> pd.DataFrame:
    return pd.DataFrame([inventory_row(p) for p in pig_ids]).set_index("pig")


if __name__ == "__main__":
    inv = build_inventory()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / "inventory.csv"
    inv.to_csv(out)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(inv.drop(columns=[c for c in inv if c.startswith("nan_%")]))
    print(f"\nWrote {out}")
