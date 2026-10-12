"""Step 5: beat-level quality control (spec §7).

Beat level (the boundary itself) → beat_ok (K,):
    rr outside [RR_MIN_S, RR_MAX_S], or differing from the median of the surrounding
    beats by more than RR_REL_DEV (ectopic beat, missed or extra detection).

Signal level (each signal within each beat) → avail (K, 6):
    more than MAX_INVALID_FRAC of the beat's samples invalid (Step 2), or a
    signal-specific plausibility check fails:
      P_ao, P_fem : pulse pressure outside [PP_MIN, PP_MAX]
      Q_ao        : stroke volume ≤ 0, or outside [SV_REL_MIN, SV_REL_MAX] × local median
      P_lv        : peak systolic P_lv below diastolic P_ao (no ejection possible)
      V_lv        : end-diastolic volume outside [EDV_REL_MIN, EDV_REL_MAX] × local median
                    (catheter jump)
      SHAPE_SIGNALS: beat shape correlating < SHAPE_MIN_CORR with the median shape of the
                    surrounding beats (erratic catheter, ringing, pulseless drift; not in the spec)
    avail is False wherever beat_ok is False.

Reasons are kept as bitfields so every rejection can be traced.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .config import CARDIAC, FS, SIGNALS
from .resample import resample_beats

RR_MIN_S, RR_MAX_S = 0.30, 1.60     # spec: 1.2 s; widened for Pig5's bradycardia (≈1.3 s)
RR_NEIGHBORS = 10                    # rr compared with the median of the 10 surrounding beats
RR_REL_DEV = 0.25
MAX_INVALID_FRAC = 0.05
PP_MIN, PP_MAX = 5.0, 100.0          # mmHg
LOCAL_BEATS = 31                     # window (beats) of the local median for SV / EDV
SV_REL_MIN, SV_REL_MAX = 0.5, 2.0
EDV_REL_MIN, EDV_REL_MAX = 0.75, 1.33
SHAPE_SIGNALS = ("P_ao", "Q_ao", "P_fem", "P_lv", "V_lv")   # not P_vc: pulse too small for a stable shape
SHAPE_MIN_CORR = 0.7

BEAT_BIT = {"rr_range": 1, "rr_irregular": 2}
SIG_BIT = {"invalid_samples": 1, "pulse_pressure": 2, "stroke_volume": 4,
           "no_ejection": 8, "edv_jump": 16, "beat_rejected": 32, "shape": 64}


def _neighbor_median(v: np.ndarray, n: int, exclude_self: bool = False) -> np.ndarray:
    """Centered rolling median over n (+1) beats, NaN-aware."""
    s = pd.Series(v)
    med = s.rolling(n + 1, center=True, min_periods=n // 2).median()
    if exclude_self:  # median of the neighbours only (spec: "the 10 surrounding beats")
        med = pd.concat([s.shift(k) for k in [*range(-n // 2, 0), *range(1, n // 2 + 1)]], axis=1).median(axis=1)
    return med.to_numpy()


def shape_corr(beats: np.ndarray, n: int = LOCAL_BEATS) -> np.ndarray:
    """Pearson correlation of each beat (K, L) with the median beat of the n surrounding beats."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN beats → NaN (already rejected)
        filled = np.where(np.isnan(beats), np.nanmedian(beats), beats)
        template = median_filter(filled, size=(n, 1), mode="nearest")
        a = beats - np.nanmean(beats, axis=1, keepdims=True)
        t = template - template.mean(axis=1, keepdims=True)
        return np.nansum(a * t, axis=1) / np.sqrt(np.nansum(a ** 2, axis=1) * np.sum(t ** 2, axis=1))


def per_beat(filtered: np.ndarray, valid: np.ndarray, b: np.ndarray, signals: list[str] = SIGNALS) -> dict:
    """Per-beat summary quantities, computed with reduceat over the boundaries b (K+1,)."""
    starts, n = b[:-1], np.diff(b)
    out = {"invalid_frac": {}, "max": {}, "min": {}}
    for s in CARDIAC:
        x, ok = filtered[signals.index(s)], valid[signals.index(s)]
        seg_x = x[b[0]:b[-1]]
        out["invalid_frac"][s] = np.add.reduceat(~ok[b[0]:b[-1]], starts - b[0]) / n
        out["max"][s] = np.maximum.reduceat(np.where(np.isnan(seg_x), -np.inf, seg_x), starts - b[0])
        out["min"][s] = np.minimum.reduceat(np.where(np.isnan(seg_x), np.inf, seg_x), starts - b[0])
    q = filtered[signals.index("Q_ao"), b[0]:b[-1]] * 1000 / 60            # L/min → mL/s
    out["sv_mL"] = np.add.reduceat(np.nan_to_num(q), starts - b[0]) / FS
    out["pp"] = {s: out["max"][s] - out["min"][s] for s in ("P_ao", "P_fem")}
    out["edv_mL"] = out["max"]["V_lv"]                                       # filling peak of the beat
    return out


def beat_qc(filtered: np.ndarray, valid: np.ndarray, boundaries: np.ndarray,
            signals: list[str] = SIGNALS) -> dict:
    b = boundaries
    rr = np.diff(b) / FS
    K = len(rr)
    f = per_beat(filtered, valid, b, signals)

    # --- 7.1 beat level -----------------------------------------------------------
    beat_flags = np.zeros(K, dtype=np.uint8)
    beat_flags[(rr < RR_MIN_S) | (rr > RR_MAX_S)] |= BEAT_BIT["rr_range"]
    rr_ref = _neighbor_median(np.where((rr >= RR_MIN_S) & (rr <= RR_MAX_S), rr, np.nan),
                              RR_NEIGHBORS, exclude_self=True)
    with np.errstate(invalid="ignore"):
        beat_flags[np.abs(rr / rr_ref - 1) > RR_REL_DEV] |= BEAT_BIT["rr_irregular"]
    beat_ok = beat_flags == 0

    # --- 7.2 signal level ---------------------------------------------------------
    sig_flags = np.zeros((K, len(CARDIAC)), dtype=np.uint8)
    col = {s: j for j, s in enumerate(CARDIAC)}
    for s in CARDIAC:
        sig_flags[f["invalid_frac"][s] > MAX_INVALID_FRAC, col[s]] |= SIG_BIT["invalid_samples"]

    with np.errstate(invalid="ignore"):
        for s in ("P_ao", "P_fem"):
            pp = f["pp"][s]
            sig_flags[(pp < PP_MIN) | (pp > PP_MAX), col[s]] |= SIG_BIT["pulse_pressure"]

        sv = f["sv_mL"]
        sv_ref = _neighbor_median(np.where(sv > 0, sv, np.nan), LOCAL_BEATS)
        bad_sv = (sv <= 0) | (sv < SV_REL_MIN * sv_ref) | (sv > SV_REL_MAX * sv_ref)
        sig_flags[bad_sv, col["Q_ao"]] |= SIG_BIT["stroke_volume"]

        sig_flags[f["max"]["P_lv"] < f["min"]["P_ao"], col["P_lv"]] |= SIG_BIT["no_ejection"]

        edv = np.where(np.isfinite(f["edv_mL"]), f["edv_mL"], np.nan)
        edv_ref = _neighbor_median(edv, LOCAL_BEATS)
        bad_edv = (edv < EDV_REL_MIN * edv_ref) | (edv > EDV_REL_MAX * edv_ref)
        sig_flags[bad_edv, col["V_lv"]] |= SIG_BIT["edv_jump"]

        for s in SHAPE_SIGNALS:
            r = shape_corr(resample_beats(filtered[signals.index(s)], b))
            sig_flags[r < SHAPE_MIN_CORR, col[s]] |= SIG_BIT["shape"]

    sig_flags[~beat_ok] |= SIG_BIT["beat_rejected"]
    avail = sig_flags == 0

    return {"beat_ok": beat_ok, "avail": avail, "beat_flags": beat_flags, "sig_flags": sig_flags,
            "sv_mL": sv.astype(np.float32), "pp_P_ao": f["pp"]["P_ao"].astype(np.float32),
            "edv_mL": f["edv_mL"].astype(np.float32)}


def summarize(qc: dict) -> pd.Series:
    """% of beats ok, and % of beats where each signal is available."""
    out = {"beat_ok": 100 * qc["beat_ok"].mean()}
    out |= {f"avail_{s}": 100 * qc["avail"][:, j].mean() for j, s in enumerate(CARDIAC)}
    return pd.Series(out)
