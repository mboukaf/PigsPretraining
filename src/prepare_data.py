"""Convert the raw OSF .mat recordings of the useful pigs into one HDF5 file.

Output layout (data/processed/pigs_useful.h5):

    attrs: fs, channels, units, bounds, flat_min_s, bound_margin_s, source, created
    pig4/
        signals  float32 (n_channels, n_samples)   channels in `CHANNELS` order
        valid    bool    (n_channels, n_samples)   False on NaN / out-of-bounds / flat segments
        attrs: paper_id, n_samples, duration_min, valid_fraction
    pig6/ ... pig7/ ... pig8/ ...

Signals are stored raw (no filtering, no normalisation). Windowing and
normalisation happen in the training Dataset, because they depend on
hyperparameters and on the train/test split.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --pigs 4 6 7 8 --out data/processed/pigs_useful.h5
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_dilation

ROOT    = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "osfstorage-archive"
OUT     = ROOT / "data" / "processed" / "pigs_useful.h5"

FS = 250  # Hz, Cushway et al. PLOS ONE 2024 (Docs/pone.0306925.pdf)

# Pigs used in the paper, with the paper's own numbering
USEFUL_PIGS = {4: 1, 6: 2, 7: 3, 8: 4}

CHANNELS = [
    "aorticPressure", "aorticFlow", "femoralPressure",
    "leftVentriclePressure", "leftVentricleVolume", "venaCavaPressure",
    "airwayPressure", "airwayFlow",
]
UNITS = {
    "aorticPressure": "mmHg", "aorticFlow": "L/min", "femoralPressure": "mmHg",
    "leftVentriclePressure": "mmHg", "leftVentricleVolume": "mL", "venaCavaPressure": "mmHg",
    "airwayPressure": "cmH2O", "airwayFlow": "mL/s",
}

# Physiological plausibility bounds; samples outside are artifacts
BOUNDS = {
    "aorticPressure":        (0, 250),
    "aorticFlow":            (-30, 60),
    "femoralPressure":       (0, 250),
    "leftVentriclePressure": (-20, 250),
    "leftVentricleVolume":   (0, 350),
    "venaCavaPressure":      (-10, 40),
    "airwayPressure":        (-5, 60),
    "airwayFlow":            (-1500, 1500),
}
BOUND_MARGIN_S = 0.5  # also invalidate this much around each out-of-bounds sample (spike ramps)
FLAT_MIN_S     = 1.0  # constant for at least this long = stuck/disconnected sensor


def load_pig(pig_id: int) -> dict[str, np.ndarray]:
    mat = sio.loadmat(RAW_DIR / f"Pig{pig_id}.mat", squeeze_me=True, struct_as_record=False)
    s = mat[f"Pig{pig_id}"]
    return {name: np.asarray(getattr(s, name), dtype=np.float32) for name in s._fieldnames}


def flat_mask(x: np.ndarray, min_len: int) -> np.ndarray:
    """True on samples belonging to a constant run of at least `min_len` samples."""
    same = np.concatenate([[False], np.diff(x) == 0, [False]]).astype(np.int8)
    d = np.diff(same)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # runs of equal diffs
    mask = np.zeros(len(x), dtype=bool)
    for s, e in zip(starts, ends):
        if e - s + 1 >= min_len:            # e - s equal steps -> e - s + 1 equal samples
            mask[s:e + 1] = True
    return mask


def valid_mask(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    nan = np.isnan(x)
    with np.errstate(invalid="ignore"):
        oob = (x < lo) | (x > hi)
    if oob.any():
        oob = binary_dilation(oob, iterations=int(BOUND_MARGIN_S * FS))
    flat = flat_mask(x, int(FLAT_MIN_S * FS))
    return ~(nan | oob | flat)


def trim_trailing_nan(signals: np.ndarray) -> np.ndarray:
    """Drop trailing samples where any channel is NaN (e.g. Pig8's last sample)."""
    finite = np.isfinite(signals).all(axis=0)
    last = len(finite) - np.argmax(finite[::-1])    # index after the last fully finite sample
    return signals[:, :last]


def main(pig_ids: list[int], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")

    with h5py.File(tmp, "w") as f:
        f.attrs["fs"] = FS
        f.attrs["channels"] = CHANNELS
        f.attrs["units"] = [UNITS[c] for c in CHANNELS]
        f.attrs["bounds"] = json.dumps(BOUNDS)
        f.attrs["bound_margin_s"] = BOUND_MARGIN_S
        f.attrs["flat_min_s"] = FLAT_MIN_S
        f.attrs["source"] = "OSF archive, Cushway et al. PLOS ONE 2024 (doi:10.1371/journal.pone.0306925)"
        f.attrs["created"] = datetime.now().isoformat(timespec="seconds")

        for pid in pig_ids:
            raw = load_pig(pid)
            missing = set(CHANNELS) - set(raw)
            if missing:
                raise KeyError(f"Pig{pid} is missing channels {missing}")

            signals = trim_trailing_nan(np.stack([raw[c] for c in CHANNELS]))
            valid = np.stack([valid_mask(signals[k], *BOUNDS[c]) for k, c in enumerate(CHANNELS)])

            g = f.create_group(f"pig{pid}")
            chunks = (len(CHANNELS), min(FS * 60, signals.shape[1]))   # one minute per chunk
            g.create_dataset("signals", data=signals, chunks=chunks,
                             compression="gzip", compression_opts=4, shuffle=True)
            g.create_dataset("valid", data=valid, chunks=chunks, compression="gzip")
            g.attrs["paper_id"] = USEFUL_PIGS.get(pid, -1)
            g.attrs["n_samples"] = signals.shape[1]
            g.attrs["duration_min"] = signals.shape[1] / FS / 60
            g.attrs["valid_fraction"] = valid.mean(axis=1)

            print(f"Pig{pid}: {signals.shape[1]:,} samples ({signals.shape[1] / FS / 60:.1f} min)")
            for c, v in zip(CHANNELS, valid.mean(axis=1)):
                print(f"    {c:22s} valid {100 * v:6.2f} %")

    tmp.replace(out)  # atomic: `out` is either the old file or the complete new one
    print(f"\nWrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pigs", type=int, nargs="+", default=list(USEFUL_PIGS))
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    main(args.pigs, args.out)
