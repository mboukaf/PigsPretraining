"""Post-hoc probe evaluation of saved encoders, including reduced-input probes (spec §15.1).

Full input: embedding from every available signal (the quick probe of training).
Reduced input: embedding computed with some signals hidden, targets still computed from the
full data. Predicting a target from an embedding that never saw its source signal requires
cross-signal relations, which an untrained encoder cannot provide — this is the test that
separates a trained encoder from the "untrained encoder" baseline (spec §15.5).

Regimes and targets:
    full        heart_period, mean_P_ao, pp_P_ao, mean_Q_ao
    no_Q_ao     mean_Q_ao                        (flow from pressures + volume)
    P_ao_only   mean_Q_ao, pp_P_ao, heart_period (bedside-like: one arterial line)

Usage (from src/):
    python -m model.evaluate E1 E2 E3                 # validation pig of each fold
    python -m model.evaluate E7 --split test          # held-out test pig of each fold
Probes are always fitted on the training pigs' windows; scores on the chosen split.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from preprocessing.config import CARDIAC, ROOT

from data.folds import make_fold
from data.store import select_recordings
from model.losses import SSLModel
from model.monitor import embed, probe_r2

REGIMES = {
    "full": (CARDIAC, ("heart_period", "mean_P_ao", "pp_P_ao", "mean_Q_ao")),
    "no_Q_ao": ([s for s in CARDIAC if s != "Q_ao"], ("mean_Q_ao",)),
    "P_ao_only": (["P_ao"], ("mean_Q_ao", "pp_P_ao", "heart_period")),
}


@torch.no_grad()
def embed_with(encoder, ds, signals: list[str], device) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings with only `signals` given; also whether each window has any of them."""
    keep = torch.tensor([s in signals for s in CARDIAC])
    encoder.eval()
    out, has = [], []
    for b in DataLoader(ds, batch_size=256, shuffle=False):
        a = b["a"] & keep
        has.append(a.flatten(1).any(1))
        a_safe = torch.where(has[-1][:, None, None], a, b["a"])     # placeholder for windows dropped below
        out.append(encoder(b["X"].to(device), b["r"].to(device), a_safe.to(device))["e"].cpu())
    return torch.cat(out).numpy(), torch.cat(has).numpy()


def evaluate_encoder(encoder, fold, device, regimes: dict = REGIMES, split: str = "val") -> dict:
    target_ds = getattr(fold, split)
    tr_full, va_full = embed(encoder, fold.train, device), embed(encoder, target_ds, device)
    res = {}
    for regime, (signals, targets) in regimes.items():
        etr, mtr = embed_with(encoder, fold.train, signals, device)
        eva, mva = embed_with(encoder, target_ds, signals, device)
        # windows without any of the kept signals have no meaningful embedding: drop them
        tr = {k: v[mtr] for k, v in (tr_full | {"e": etr}).items()}
        va = {k: v[mva] for k, v in (va_full | {"e": eva}).items()}
        r2 = probe_r2(tr, va)
        res |= {f"{regime}:{t}": r2[t] for t in targets}
        res |= {f"{regime}:{t}@within": r2[f"{t}@within"] for t in targets}
    return res


def main(runs: list[str], split: str = "val"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []
    for run in runs:
        for ckpt_path in sorted((ROOT / "runs" / run).glob("fold*/encoder_best.pt")):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            args = ckpt["args"]
            recs = select_recordings(args.get("pigs"), args.get("no_flow_pigs"))   # the run's own data
            fold = make_fold(recs, args["test_pig"], args.get("val_pig"),
                             vlv_mode=args.get("vlv_mode", "relative"), qao_mode=args.get("qao_mode", "zero"))
            torch.manual_seed(args.get("seed", 0))                 # same init as the run's epoch 0
            untrained = SSLModel().encoder.to(device)
            trained = SSLModel().encoder.to(device)
            trained.load_state_dict(ckpt["encoder"])
            for name, enc in (("untrained", untrained), (f"trained (ep {ckpt['epoch']})", trained)):
                rows.append({"run": run, "fold": ckpt_path.parent.name, "split": split, "encoder": name} |
                            evaluate_encoder(enc, fold, device, split=split))
    df = pd.DataFrame(rows)
    out = ROOT / "runs" / "evaluate_summary.json"
    prev = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(prev + rows, indent=1))
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:+.2f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    split = "val"
    if "--split" in args:
        i = args.index("--split")
        split = args[i + 1]
        del args[i:i + 2]
    main(args, split)
