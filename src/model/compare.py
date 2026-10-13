"""Summarize training runs: runs/<run>/fold<k>/log.json → one row per (run, fold).

Usage (from src/):
    python -m model.compare E0 E1 E2
"""
import json
import sys

import numpy as np
import pandas as pd

from preprocessing.config import ROOT

from model.monitor import PROBE_TARGETS


def summarize(run: str) -> pd.DataFrame:
    rows = []
    for log_path in sorted((ROOT / "runs" / run).glob("fold*/log.json")):
        log = [r for r in json.loads(log_path.read_text()) if "probe_mean" in r]
        trained = [r for r in log if r["epoch"] > 0]
        if not trained:
            continue
        best = max(trained, key=lambda r: r["probe_mean"])      # best TRAINED epoch (= saved checkpoint)
        first, last = log[0], log[-1]
        row = {"run": run, "fold": log_path.parent.name,
               "untrained": first["probe_mean"] if first["epoch"] == 0 else np.nan,
               "best": best["probe_mean"], "best_ep": best["epoch"], "last": last["probe_mean"]}
        within = "probe_heart_period@within" in best
        sfx = "@within" if within else ""
        row["metric"] = "within-pig R²" if within else "R²"
        for k in [*PROBE_TARGETS, "P_ao_only:mean_Q_ao", "no_Q_ao:mean_Q_ao"]:
            if f"probe_{k}{sfx}" in best:
                row[f"untrained_{k}"] = first.get(f"probe_{k}{sfx}", np.nan)
                row[f"best_{k}"] = best[f"probe_{k}{sfx}"]
        row |= {"rank_val_best": best["eff_rank_val"], "rank_val_last": last["eff_rank_val"],
                "valA_best": best.get("val_A", np.nan), "valA_last": last.get("val_A", np.nan)}
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.concat([summarize(r) for r in sys.argv[1:]], ignore_index=True)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
