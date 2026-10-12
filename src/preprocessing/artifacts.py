"""Step 2: artifact detection (spec §4). Artifacts are marked, never repaired.

Each rule returns a boolean mask (True = artifact) at the raw sampling rate.
The rules are stored as a bitfield so we can see which rule fired where:

    flags[k, t] & BIT["flat"]  -> rule "flat" fired on signal k at sample t

and the final mask is

    valid = NOT dilate(flags != 0, margin)      (spikes excluded, see rule_spike)

Thresholds were chosen from the data (percentiles and rolling-std distributions
over all 8 pigs), not from the spec's default ranges, which don't match these animals.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .config import FS, SIGNALS

# --- rule parameters --------------------------------------------------------------

BIT = {"nan": 1, "flat": 2, "saturation": 4, "flush": 8,
       "out_of_range": 16, "damping": 32, "manual": 64, "spike": 128, "no_pulse": 256}
NOT_INVALIDATING = BIT["spike"]  # spikes are removed by the median prefilter of Step 3

BOUNDS = {                       # plausible range per signal (outside = artifact)
    "P_ao":  (5, 140),           # mmHg
    "Q_ao":  (-20, 50),          # L/min
    "P_fem": (5, 140),           # mmHg
    "P_lv":  (-15, 150),         # mmHg
    "V_lv":  (-0.5, 290),        # mL; clipping at exactly 0 is caught by the saturation rule
    "P_vc":  (-5, 30),           # mmHg
    "P_aw":  (-5, 70),           # cmH2O
    "Q_aw":  (-1000, 700),       # mL/s
}
FLAT_WIN_S = 1.0                 # rolling-std window
FLAT_STD = {s: 0.05 for s in SIGNALS} | {"P_vc": 0.01, "P_aw": 0.01, "Q_aw": 0.01}
NO_PULSE_WIN_S = 2.0             # longer than one beat
NO_PULSE_MIN = {"P_ao": 5.0, "P_fem": 5.0, "P_lv": 5.0, "Q_ao": 0.3, "V_lv": 3.0}  # peak-to-peak
SAT_MIN_S = 0.02                 # stuck at the recording's min/max for at least 20 ms
FLUSH_SIGNALS = ["P_ao", "P_fem", "P_lv"]
FLUSH_LEVEL, FLUSH_MIN_S = 200.0, 0.5
DAMP_SIGNALS = ("P_ao", "P_fem")  # damping is checked between the two arterial pressures
DAMP_PP_WIN_S = 1.5              # pulse pressure = max - min over this window (> 1 beat)
DAMP_REF_S = 60.0                # reference pulse pressure = rolling median over this window
DAMP_DROP, DAMP_OTHER_OK = 0.5, 0.8
SPIKE_MAX_SAMPLES = 2            # out-of-range runs this short are spikes, not dropouts
MARGIN_S = 0.5                   # every artifact run is extended by this on each side

# Whole-recording exclusions decided in Step 1
MANUAL = {
    1: {"Q_ao": "rectified (never negative, 50% exact zeros)",
        "Q_aw": "expiration missing (never negative, 44% NaN)"},
    3: {"V_lv": "abnormal but consistent shape (flat ~15 mL with 100-250 mL bursts), "
                "non-physiological PV loops, EDV variability 3-8x other pigs"},
}


# --- helpers ----------------------------------------------------------------------

def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, lengths) of the True runs of a boolean array."""
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    return starts, np.flatnonzero(d == -1) - starts


def _keep_long_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros_like(mask)
    for s, n in zip(*_runs(mask)):
        if n >= min_len:
            out[s:s + n] = True
    return out


def _fill_nan(x: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), np.nanmedian(x), x)


# --- rules ------------------------------------------------------------------------

def rule_nan(x: np.ndarray, sig: str) -> np.ndarray:
    return np.isnan(x)


def rule_flat(x: np.ndarray, sig: str) -> np.ndarray:
    """Rolling standard deviation over FLAT_WIN_S below a per-signal threshold."""
    xf = _fill_nan(x)
    n = int(FLAT_WIN_S * FS)
    mean = uniform_filter1d(xf, n)
    var = uniform_filter1d(xf ** 2, n) - mean ** 2
    return np.sqrt(np.clip(var, 0, None)) < FLAT_STD[sig]


def rule_no_pulse(x: np.ndarray, sig: str) -> np.ndarray:
    """Peak-to-peak amplitude over NO_PULSE_WIN_S below a per-signal minimum:
    a disconnected transducer that drifts or is noisy but shows no heartbeat."""
    if sig not in NO_PULSE_MIN:
        return np.zeros(len(x), dtype=bool)
    xf = _fill_nan(x)
    n = int(NO_PULSE_WIN_S * FS)
    return maximum_filter1d(xf, n) + maximum_filter1d(-xf, n) < NO_PULSE_MIN[sig]


def rule_saturation(x: np.ndarray, sig: str) -> np.ndarray:
    """Stuck exactly at the recording's minimum or maximum for at least SAT_MIN_S."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    return _keep_long_runs((x == lo) | (x == hi), int(SAT_MIN_S * FS))


def rule_flush(x: np.ndarray, sig: str) -> np.ndarray:
    """Pressure line flush: above FLUSH_LEVEL for at least FLUSH_MIN_S."""
    if sig not in FLUSH_SIGNALS:
        return np.zeros(len(x), dtype=bool)
    with np.errstate(invalid="ignore"):
        return _keep_long_runs(x > FLUSH_LEVEL, int(FLUSH_MIN_S * FS))


def _outside(x: np.ndarray, sig: str) -> np.ndarray:
    lo, hi = BOUNDS[sig]
    with np.errstate(invalid="ignore"):
        return (x < lo) | (x > hi)


def rule_out_of_range(x: np.ndarray, sig: str) -> np.ndarray:
    """Outside BOUNDS for longer than a spike."""
    return _keep_long_runs(_outside(x, sig), SPIKE_MAX_SAMPLES + 1)


def rule_spike(x: np.ndarray, sig: str) -> np.ndarray:
    """Outside BOUNDS for at most SPIKE_MAX_SAMPLES samples (electrical spike)."""
    out = _outside(x, sig)
    return out & ~_keep_long_runs(out, SPIKE_MAX_SAMPLES + 1)


def _pulse_pressure_ratio(x: np.ndarray) -> np.ndarray:
    """Local pulse pressure divided by its rolling median over DAMP_REF_S."""
    xf = _fill_nan(x)
    n = int(DAMP_PP_WIN_S * FS)
    pp = maximum_filter1d(xf, n) - (-maximum_filter1d(-xf, n))
    step = FS // 2                                           # evaluate the median at 2 Hz
    ref = pd.Series(pp[::step]).rolling(int(DAMP_REF_S * 2), center=True, min_periods=1).median()
    ref = np.repeat(ref.to_numpy(), step)[:len(x)]
    return pp / np.maximum(ref, 1e-6)


def rule_damping(signals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pulse pressure of one arterial line drops by > 50 % while the other stays normal."""
    a, b = DAMP_SIGNALS
    ra, rb = _pulse_pressure_ratio(signals[a]), _pulse_pressure_ratio(signals[b])
    return {a: (ra < DAMP_DROP) & (rb > DAMP_OTHER_OK),
            b: (rb < DAMP_DROP) & (ra > DAMP_OTHER_OK)}


PER_SIGNAL_RULES = {
    "nan": rule_nan, "flat": rule_flat, "no_pulse": rule_no_pulse, "saturation": rule_saturation,
    "flush": rule_flush, "out_of_range": rule_out_of_range, "spike": rule_spike,
}


# --- combination ------------------------------------------------------------------

def detect_artifacts(raw: np.ndarray, pig_id: int, signals: list[str] = SIGNALS):
    """Apply all rules to one recording.

    raw : (n_signals, N) raw values, rows in `signals` order
    returns
        flags : uint16 (n_signals, N)  bitfield of the rules that fired (before the margin)
        valid : bool  (n_signals, N)  usable samples (after extending artifacts by MARGIN_S);
                                      spikes don't count, Step 3 removes them
    """
    flags = np.zeros(raw.shape, dtype=np.uint16)
    by_name = dict(zip(signals, raw))

    for k, sig in enumerate(signals):
        for name, rule in PER_SIGNAL_RULES.items():
            flags[k][rule(raw[k], sig)] |= BIT[name]

    if all(s in by_name for s in DAMP_SIGNALS):
        for sig, m in rule_damping(by_name).items():
            flags[signals.index(sig)][m] |= BIT["damping"]

    for sig in MANUAL.get(pig_id, {}):
        if sig in by_name:
            flags[signals.index(sig)] |= BIT["manual"]

    margin = 2 * int(MARGIN_S * FS) + 1
    bad = (flags & ~np.uint16(NOT_INVALIDATING)) != 0
    valid = ~maximum_filter1d(bad.view(np.uint8), margin, axis=1).astype(bool)
    return flags, valid


def summarize(flags: np.ndarray, valid: np.ndarray, signals: list[str] = SIGNALS) -> pd.DataFrame:
    """% of samples flagged by each rule, and % valid after the margin, per signal."""
    rows = {}
    for k, sig in enumerate(signals):
        rows[sig] = {name: 100 * np.mean(flags[k] & bit != 0) for name, bit in BIT.items()}
        rows[sig]["valid"] = 100 * valid[k].mean()
    return pd.DataFrame(rows).T
