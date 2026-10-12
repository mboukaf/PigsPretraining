"""Resampling beats to a fixed number of points (spec §8, Step 6). Also used by Step 5's
shape check.

Beat i spans samples [b_i, b_{i+1}). It is resampled to L points on the uniform grid
    b_i + n_i · j / L,   j = 0 … L−1,   n_i = b_{i+1} − b_i
by linear interpolation. NaN inside a beat propagates to the points that use it.
"""
import numpy as np

L_POINTS = 64


def resample_beats(x: np.ndarray, boundaries: np.ndarray, L: int = L_POINTS) -> np.ndarray:
    """x: (N,) signal, boundaries: (K+1,) → (K, L) beats. Fully vectorised."""
    b = boundaries.astype(np.float64)
    pos = b[:-1, None] + np.diff(b)[:, None] * (np.arange(L) / L)[None, :]      # (K, L) fractional samples
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    i1 = np.minimum(i0 + 1, len(x) - 1)
    return x[i0] * (1 - frac) + x[i1] * frac


def resample_recording(filtered: np.ndarray, boundaries: np.ndarray, avail: np.ndarray,
                       signals: list[str], cardiac: list[str], L: int = L_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """All cardiac signals of one recording → beats (K, M, L) float32 and updated avail (K, M).

    Entries whose resampled beat contains NaN (a few invalid samples, allowed by Step 5's
    5 % rule) become unavailable. Unavailable entries are filled with 0: they are never
    used, the availability mask tells the encoder they are missing (spec §8).
    """
    X = np.stack([resample_beats(filtered[signals.index(s)], boundaries, L) for s in cardiac], axis=1)
    avail = avail & ~np.isnan(X).any(axis=2)
    X[~avail] = 0.0
    return X.astype(np.float32), avail
