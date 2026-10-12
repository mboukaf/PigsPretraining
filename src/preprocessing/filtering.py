"""Step 3: filtering (spec §5).

1. Despike: samples flagged as `spike` in Step 2 are replaced by the median of their
   5-sample neighbourhood. Only those samples change; real sharp features (upstroke,
   dicrotic notch) are untouched.
2. Low-pass: zero-phase (forward-backward) 4th-order Butterworth, applied separately to
   each valid stretch so that artifacts cannot leak into neighbouring good data.
   Invalid samples are NaN in the output. No high-pass: the DC level is information.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from .artifacts import BIT, _runs
from .config import FS, SIGNALS

ORDER = 4
# Spec default 25 Hz. V_lv: ~50 % of its power lies above 25 Hz and is flat up to
# 125 Hz (broadband catheter noise); 10 Hz gives clean PV loops without changing their shape.
CUTOFF_HZ = {s: 25.0 for s in SIGNALS} | {"V_lv": 10.0}
DESPIKE_WIDTH = 5


def despike(x: np.ndarray, spike: np.ndarray) -> np.ndarray:
    """Replace flagged samples by the local median (computed without NaN propagation)."""
    if not spike.any():
        return x
    xf = np.where(np.isnan(x), np.nanmedian(x), x)
    out = x.copy()
    out[spike] = median_filter(xf, DESPIKE_WIDTH)[spike]
    return out


def lowpass_valid(x: np.ndarray, valid: np.ndarray, cutoff: float) -> np.ndarray:
    """Zero-phase low-pass on each valid stretch; NaN elsewhere.

    Stretches too short for the filter's edge padding are left NaN (they are a few
    samples long and would not hold a single beat anyway).
    """
    sos = signal.butter(ORDER, cutoff, fs=FS, output="sos")
    padlen = 3 * (2 * len(sos) + 1)            # sosfiltfilt's default padding
    out = np.full(len(x), np.nan)
    for s, n in zip(*_runs(valid)):
        if n > padlen:
            out[s:s + n] = signal.sosfiltfilt(sos, x[s:s + n])
    return out


def filter_recording(raw: np.ndarray, flags: np.ndarray, valid: np.ndarray,
                     signals: list[str] = SIGNALS) -> np.ndarray:
    """raw, flags, valid: (n_signals, N) → filtered float64 (n_signals, N), NaN where invalid."""
    out = np.empty(raw.shape)
    for k, sig in enumerate(signals):
        x = despike(raw[k].astype(np.float64), flags[k] & BIT["spike"] != 0)
        out[k] = lowpass_valid(x, valid[k], CUTOFF_HZ[sig])
    return out
