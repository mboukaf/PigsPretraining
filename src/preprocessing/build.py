"""Run the preprocessing steps and store their outputs in data/processed/pigs.h5.

Layout (grows as steps are added):

    attrs: fs, signals, units, cardiac, created, step2 … step5 (JSON parameters)
    pig{p}/
        attrs: paper_id (-1 if not in the paper), n_samples
        raw    float32 (8, N)   raw values, rows in `signals` order        (Step 1)
        flags  uint16  (8, N)   bitfield of artifact rules, see artifacts.BIT (Step 2)
        valid  bool    (8, N)   usable samples                             (Step 2)
        filtered float32 (8, N) despiked + zero-phase low-pass, NaN where invalid (Step 3)
        beats/                                                             (Step 4)
            boundaries int64 (K+1,)  sample index of each beat start (+ end of last beat)
            rr         float32 (K,)  beat durations [s]
            t_beat     float64 (K,)  beat start times [s]
            grid_src   uint8 (K,)    index into attrs grid_sources of the defining signal
            attrs: grid_sources, offsets_ms (fallback upstroke − P_ao boundary)
            beat_ok    bool (K,)     beat kept at all                          (Step 5)
            avail      bool (K, 6)   signal usable in beat, columns = attrs cardiac
            beat_flags uint8 (K,)    reasons, see qc.BEAT_BIT
            sig_flags  uint8 (K, 6)  reasons, see qc.SIG_BIT
            sv_mL, pp_P_ao, edv_mL float32 (K,)  per-beat quantities used by the checks
            X          float32 (K, 6, 64)  beats resampled to 64 points, 0 where unavailable (Step 6)
            avail      is updated by Step 6 (resampled beats containing NaN become unavailable)

Usage (from src/):
    python -m preprocessing.build
    python -m preprocessing.build --pigs 4 6
"""
import argparse
import json
from datetime import datetime

import h5py
import numpy as np

from . import artifacts, beats, filtering, qc, resample
from .config import ALL_PIGS, CARDIAC, FS, PAPER_ID, PROC_DIR, ROOT, SIGNALS, UNITS
from .io import load_raw
from .plots import plot_beat_overlay, plot_resampled, plot_validity

OUT = PROC_DIR / "pigs.h5"
FIG_DIR = ROOT / "figures" / "preprocessing"


def step2_params() -> dict:
    return {
        "bits": artifacts.BIT, "bounds": artifacts.BOUNDS, "flat_win_s": artifacts.FLAT_WIN_S,
        "flat_std": artifacts.FLAT_STD,
        "no_pulse_win_s": artifacts.NO_PULSE_WIN_S, "no_pulse_min": artifacts.NO_PULSE_MIN,
        "sat_min_s": artifacts.SAT_MIN_S,
        "flush_level": artifacts.FLUSH_LEVEL, "flush_min_s": artifacts.FLUSH_MIN_S,
        "damping": {"signals": artifacts.DAMP_SIGNALS, "pp_win_s": artifacts.DAMP_PP_WIN_S,
                    "ref_s": artifacts.DAMP_REF_S, "drop": artifacts.DAMP_DROP,
                    "other_ok": artifacts.DAMP_OTHER_OK},
        "spike_max_samples": artifacts.SPIKE_MAX_SAMPLES, "margin_s": artifacts.MARGIN_S,
        "manual": {str(k): v for k, v in artifacts.MANUAL.items()},
    }


def step3_params() -> dict:
    return {"order": filtering.ORDER, "cutoff_hz": filtering.CUTOFF_HZ,
            "despike_width": filtering.DESPIKE_WIDTH, "zero_phase": True}


def step4_params() -> dict:
    return {"grid_sources": beats.GRID_SOURCES, "delta_shift_s": beats.DELTA_SHIFT_S,
            "min_beat_s": beats.MIN_BEAT_S, "local_win_s": beats.LOCAL_WIN_S,
            "rel_height": beats.REL_HEIGHT, "foot_search_s": beats.FOOT_SEARCH_S,
            "rise_after_s": beats.RISE_AFTER_S, "pp_win_s": beats.PP_WIN_S, "min_rise": beats.MIN_RISE}


def step5_params() -> dict:
    return {"rr_s": [qc.RR_MIN_S, qc.RR_MAX_S], "rr_neighbors": qc.RR_NEIGHBORS, "rr_rel_dev": qc.RR_REL_DEV,
            "max_invalid_frac": qc.MAX_INVALID_FRAC, "pp_mmHg": [qc.PP_MIN, qc.PP_MAX],
            "local_beats": qc.LOCAL_BEATS, "sv_rel": [qc.SV_REL_MIN, qc.SV_REL_MAX],
            "edv_rel": [qc.EDV_REL_MIN, qc.EDV_REL_MAX], "shape_signals": qc.SHAPE_SIGNALS,
            "shape_min_corr": qc.SHAPE_MIN_CORR, "beat_bits": qc.BEAT_BIT, "sig_bits": qc.SIG_BIT}


def main(pig_ids: list[int]) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".h5.tmp")

    with h5py.File(tmp, "w") as f:
        f.attrs["fs"] = FS
        f.attrs["signals"] = SIGNALS
        f.attrs["units"] = [UNITS[s] for s in SIGNALS]
        f.attrs["cardiac"] = CARDIAC
        f.attrs["created"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["step2"] = json.dumps(step2_params())
        f.attrs["step3"] = json.dumps(step3_params())
        f.attrs["step4"] = json.dumps(step4_params())
        f.attrs["step5"] = json.dumps(step5_params())
        f.attrs["step6"] = json.dumps({"L": resample.L_POINTS, "method": "linear",
                                       "grid": "b_i + n_i * j / L, j = 0..L-1", "unavailable_fill": 0.0})

        for p in pig_ids:
            raw = load_raw(p)
            flags, valid = artifacts.detect_artifacts(raw, p)
            filtered = filtering.filter_recording(raw, flags, valid)
            valid &= ~np.isnan(filtered)    # valid islands too short to filter (< 0.1 s) become invalid

            g = f.create_group(f"pig{p}")
            g.attrs["paper_id"] = PAPER_ID.get(p, -1)
            g.attrs["n_samples"] = raw.shape[1]
            chunks = (len(SIGNALS), min(60 * FS, raw.shape[1]))     # one minute per chunk
            opts = dict(chunks=chunks, compression="gzip", compression_opts=4, shuffle=True)
            g.create_dataset("raw", data=raw.astype(np.float32), **opts)
            g.create_dataset("flags", data=flags, **opts)
            g.create_dataset("valid", data=valid, **opts)
            g.create_dataset("filtered", data=filtered.astype(np.float32), **opts)

            grid = beats.beat_grid(filtered, valid)
            gb = g.create_group("beats")
            gb.create_dataset("boundaries", data=grid["boundaries"])
            gb.create_dataset("rr", data=grid["rr"].astype(np.float32))
            gb.create_dataset("t_beat", data=grid["t_beat"])
            gb.create_dataset("grid_src", data=grid["grid_src"])
            gb.attrs["grid_sources"] = beats.GRID_SOURCES
            gb.attrs["offsets_ms"] = json.dumps(grid["offsets_ms"])

            check = qc.beat_qc(filtered, valid, grid["boundaries"])
            X, check["avail"] = resample.resample_recording(filtered, grid["boundaries"], check["avail"],
                                                            SIGNALS, CARDIAC)
            for name in ("beat_ok", "avail", "beat_flags", "sig_flags", "sv_mL", "pp_P_ao", "edv_mL"):
                gb.create_dataset(name, data=check[name])
            gb.create_dataset("X", data=X, chunks=(min(1024, len(X)), len(CARDIAC), resample.L_POINTS),
                              compression="gzip", compression_opts=4, shuffle=True)
            fig = plot_resampled(X, check["avail"], grid["rr"], p, CARDIAC)
            fig.savefig(FIG_DIR / f"step6_resampled_pig{p}.png", dpi=100, bbox_inches="tight")

            print(f"\nPig{p}: {raw.shape[1]:,} samples, valid %:")
            print(artifacts.summarize(flags, valid)["valid"].round(2).to_string())
            fig = plot_validity(raw, valid, p)
            fig.savefig(FIG_DIR / f"step2_validity_pig{p}.png", dpi=110, bbox_inches="tight")
            fig = plot_beat_overlay(filtered, grid["boundaries"], grid["grid_src"], p)
            fig.savefig(FIG_DIR / f"step4_beats_pig{p}.png", dpi=100, bbox_inches="tight")
            src = np.bincount(grid["grid_src"], minlength=len(beats.GRID_SOURCES))
            print(f"  beats: {len(grid['rr']):,} | HR {60 / np.median(grid['rr']):.1f} bpm | "
                  f"from {dict(zip(beats.GRID_SOURCES, src.tolist()))}")
            print("  QC % :", qc.summarize(check).round(1).to_dict())

    tmp.replace(OUT)
    print(f"\nWrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.0f} MB), figures in {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pigs", type=int, nargs="+", default=ALL_PIGS)
    main(ap.parse_args().pigs)
