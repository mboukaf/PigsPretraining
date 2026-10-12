"""Step 4: beat grid (spec §6). One common set of beat boundaries per recording.

1. On the reference pressure, find each beat's upstroke = peak of the derivative.
2. P_ao: foot by the intersecting-tangent method (horizontal line through the preceding
   diastolic minimum ∩ tangent at the point of maximum slope), then shifted DELTA_SHIFT_S
   earlier so the LV's isovolumic contraction stays inside the same beat.
3. Where P_ao is invalid, the grid comes from P_lv, then P_fem. Their boundary is
   (upstroke time − offset), with the offset measured on this recording where both the
   fallback signal and P_ao are valid (spec: fixed ≈20 ms for P_lv; measured here instead,
   because the femoral delay varies 64–196 ms between pigs).
4. Candidate boundaries from all sources are merged by priority and de-duplicated.

Output per recording:
    boundaries : (K+1,) int64   sample index of the start of each beat
    rr         : (K,)   float   beat durations in s
    t_beat     : (K,)   float   start time of each beat in s
    grid_src   : (K,)   uint8   index into GRID_SOURCES of the signal that defined the beat start
"""
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

from .config import FS, SIGNALS

GRID_SOURCES = ["P_ao", "P_lv", "P_fem"]   # order of preference
DELTA_SHIFT_S = 0.080                      # boundary = aortic foot − 80 ms
MIN_BEAT_S = 0.30                          # minimum distance between upstrokes (HR < 200 bpm)
LOCAL_WIN_S = 5.0                          # window of the local reference for the peak threshold
REL_HEIGHT = 0.4                           # upstroke slope must reach 40 % of the local max slope
FOOT_SEARCH_S = 0.25                       # diastolic minimum searched up to 250 ms before the upstroke
RISE_AFTER_S = 0.15                        # systolic rise measured up to 150 ms after the upstroke
PP_WIN_S = 2.0                             # local pulse pressure = max − min over 2 s
MIN_RISE = 0.5                             # a beat's rise must reach 50 % of the local pulse pressure


def upstrokes(x: np.ndarray) -> np.ndarray:
    """Sample indices of the maximum upstroke slope of each beat of a (filtered) pressure.

    Two criteria, both relative to the local signal so they follow slow changes in pulse
    amplitude (interventions, terminal phase):
      - slope: at least REL_HEIGHT of the local maximum slope over LOCAL_WIN_S;
      - amplitude: the pressure must then rise by at least MIN_RISE of the local pulse
        pressure. This rejects the rise after the dicrotic notch, which can be steep but
        only adds a few mmHg.
    """
    dx = np.gradient(x)
    dx = np.where(np.isnan(dx), -np.inf, dx)
    local_max = maximum_filter1d(np.where(np.isfinite(dx), dx, 0), int(LOCAL_WIN_S * FS))
    peaks, _ = signal.find_peaks(dx, distance=int(MIN_BEAT_S * FS))
    peaks = peaks[(dx[peaks] > 0) & (dx[peaks] >= REL_HEIGHT * local_max[peaks])]

    xf = np.where(np.isnan(x), np.nanmedian(x), x)
    n = int(PP_WIN_S * FS)
    local_pp = maximum_filter1d(xf, n) + maximum_filter1d(-xf, n)
    before, after = int(FOOT_SEARCH_S * FS), int(RISE_AFTER_S * FS)
    peaks = peaks[(peaks >= before) & (peaks < len(x) - after)]
    lo = xf[peaks[:, None] + np.arange(-before, 1)].min(axis=1)   # diastolic minimum before the upstroke
    hi = xf[peaks[:, None] + np.arange(0, after + 1)].max(axis=1)   # systolic level after it
    return peaks[hi - lo >= MIN_RISE * local_pp[peaks]]


def feet(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Intersecting-tangent foot (float sample index) before each upstroke peak; NaN if undefined."""
    dx = np.gradient(x)
    w = int(FOOT_SEARCH_S * FS)
    out = np.full(len(peaks), np.nan)
    for i, p in enumerate(peaks):
        if p < w:
            continue
        seg = x[p - w:p + 1]
        if np.isnan(seg).any() or dx[p] <= 0:
            continue
        j = np.argmin(seg)
        t_min, p_min = p - w + j, seg[j]
        t_foot = p + (p_min - x[p]) / dx[p]          # where the tangent at p reaches p_min
        out[i] = np.clip(t_foot, t_min, p)
    return out


def _source_boundaries(filtered: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Candidate boundaries (float samples) from each grid source, before merging,
    and the measured offset (ms) between each fallback's upstroke and the P_ao boundary."""
    shift = DELTA_SHIFT_S * FS
    p_ao = filtered["P_ao"]
    ao_peaks = upstrokes(p_ao)
    ao_feet = feet(p_ao, ao_peaks)
    keep = ~np.isnan(ao_feet)
    ao_peaks, ao_b = ao_peaks[keep], ao_feet[keep] - shift
    out, offsets = {"P_ao": ao_b}, {}

    for src in GRID_SOURCES[1:]:
        pk = upstrokes(filtered[src])
        if len(pk) == 0 or len(ao_peaks) == 0:
            out[src] = np.array([])
            continue
        # match each fallback upstroke to the nearest aortic upstroke (both valid) to measure the offset
        j = np.clip(np.searchsorted(ao_peaks, pk), 1, len(ao_peaks) - 1)
        nearest = np.where(np.abs(ao_peaks[j - 1] - pk) < np.abs(ao_peaks[j] - pk), j - 1, j)
        close = np.abs(ao_peaks[nearest] - pk) < 0.3 * FS
        if close.sum() < 50:                         # not enough overlap to calibrate this source
            out[src] = np.array([])
            continue
        offset = np.median(pk[close] - ao_b[nearest[close]])
        offsets[src] = float(offset / FS * 1000)
        out[src] = pk - offset
    return out, offsets


def beat_grid(filtered: np.ndarray, valid: np.ndarray, signals: list[str] = SIGNALS) -> dict:
    """filtered, valid: (n_signals, N) from Step 3 → beat grid of the recording."""
    sig = {s: filtered[signals.index(s)] for s in GRID_SOURCES}
    ok = {s: valid[signals.index(s)] for s in GRID_SOURCES}
    cand, offsets = _source_boundaries(sig)

    # each sample is owned by the highest-priority source that is valid there;
    # a source contributes only the boundaries that fall in samples it owns
    owner = np.full(valid.shape[1], -1, dtype=np.int8)
    for k, s in reversed(list(enumerate(GRID_SOURCES))):
        owner[ok[s]] = k
    bs, src = [], []
    for k, s in enumerate(GRID_SOURCES):
        b = cand[s]
        b = b[(b >= 0) & (b < len(owner))]
        b = b[owner[b.astype(int)] == k]
        bs.append(b)
        src.append(np.full(len(b), k, dtype=np.uint8))
    b, src = np.concatenate(bs), np.concatenate(src)
    order = np.argsort(b)
    b, src = b[order], src[order]

    # at source transitions two sources may mark the same beat: keep the higher-priority one
    keep = np.ones(len(b), dtype=bool)
    min_gap = MIN_BEAT_S * FS
    for i in range(1, len(b)):
        if keep[i - 1] and b[i] - b[i - 1] < min_gap:
            if src[i] < src[i - 1]:
                keep[i - 1] = False
            else:
                keep[i] = False
    b, src = np.round(b[keep]).astype(np.int64), src[keep]

    return {
        "boundaries": b,
        "rr": np.diff(b) / FS,
        "t_beat": b[:-1] / FS,
        "grid_src": src[:-1],
        "offsets_ms": offsets,
    }
