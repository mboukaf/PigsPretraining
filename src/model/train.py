"""Step 12: self-supervised training of the encoder on one leave-one-pig-out fold (spec §14).

Optimization: AdamW, lr 5e-4, linear warm-up over the first 5 % of steps, cosine decay to
1e-5, weight decay 0.05 (not on biases, norms, embeddings), gradient clipping 1.0.
Batches avoid windows of the same pig closer than MIN_GAP_S in time (spec §13.4).
Every `eval_every` epochs: loss terms on the validation pig, collapse indicators of e
(per-dimension std, effective rank) and a quick ridge probe (train pigs → validation pig).
The checkpoint kept is the one with the best validation probe score, not the lowest loss.
Score = mean of the clipped WITHIN-PIG R² of the full-input probes (4 targets) and of the
reduced-input probes (relative flow from P_ao only, and from all signals but Q_ao). The
reduced-input probes are where training must beat the untrained encoder (spec §15.1); within-pig
R² ignores constant between-pig offsets (see model/monitor.py and Docs/experiments.md, E4 / E7).
An encoder checkpoint is saved at every evaluation (encoder_ep<k>.pt) so that other selection
rules can be applied afterwards.

Usage (from src/):
    python -m model.train --test-pig 7 --epochs 10 --run-name E7
Defaults = the configuration chosen in Docs/experiments.md (E7): scaled Q_ao / V_lv, γ 0.1,
sensor-dropout keep probability 0.7.

Outputs go to runs/<run-name>/fold<test-pig>/ (log.json, normalizer.json, encoder_best.pt).
Epoch 0 is an evaluation of the untrained model (baseline "untrained encoder", spec §15.5).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from preprocessing.config import ROOT

from data.folds import make_fold
from data.store import select_recordings
from model.features import window_summaries
from model.losses import SSLModel
from model.evaluate import evaluate_encoder
from model.monitor import PROBE_TARGETS, effective_rank, embed, probe_r2
from model.views import TwoViewDataset, make_view, offset_scale

MIN_GAP_S = 60.0
NO_DECAY = ("pos", "cls", "missing", "mask_token")


class SpacedBatchSampler(Sampler):
    """Shuffled batches in which no two windows of the same pig start < min_gap_s apart.
    Windows that conflict with the current batch are deferred to later batches."""

    def __init__(self, ds, batch_size: int, min_gap_s: float = MIN_GAP_S, seed: int = 0):
        self.pig = np.array([p for p, _ in ds.index])
        self.t = np.array([ds.data[p].t_beat[s] for p, s in ds.index])
        self.bs, self.gap = batch_size, min_gap_s
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        pending = list(self.rng.permutation(len(self.pig)))
        while len(pending) >= self.bs:
            batch, times, rest = [], {}, []
            for i in pending:
                if len(batch) < self.bs and all(abs(self.t[i] - t) >= self.gap for t in times.get(self.pig[i], [])):
                    batch.append(i)
                    times.setdefault(self.pig[i], []).append(self.t[i])
                else:
                    rest.append(i)
            if len(batch) < self.bs:        # remaining windows cannot form a full spaced batch
                break
            yield batch
            pending = rest

    def __len__(self) -> int:
        return len(self.pig) // self.bs


def param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.split(".")[-1] in NO_DECAY else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_lambda(total_steps: int, warmup_frac: float, lr: float, lr_min: float):
    warm = max(1, int(warmup_frac * total_steps))
    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return (lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * prog))) / lr
    return f


SELECTION_REGIMES = {"no_Q_ao": (["P_ao", "P_fem", "P_lv", "V_lv", "P_vc"], ("mean_Q_ao",)),
                     "P_ao_only": (["P_ao"], ("mean_Q_ao",))}


def selection_score(ev: dict, within: bool = True) -> float:
    suffix = "@within" if within else ""
    keys = [f"probe_{k}{suffix}" for k in PROBE_TARGETS] + [f"probe_{r}:mean_Q_ao{suffix}" for r in SELECTION_REGIMES]
    return float(np.nanmean([max(ev[k], -1.0) for k in keys]))


def summary_stats(ds) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean / std of the window summaries over a dataset (NaN-aware), for standardizing L_S."""
    S = torch.cat([window_summaries(b["X"], b["r"], b["a"]) for b in DataLoader(ds, batch_size=512)])
    mean = torch.nanmean(S, dim=0)
    std = torch.sqrt(torch.nanmean((S - mean) ** 2, dim=0))
    return torch.nan_to_num(mean), torch.nan_to_num(std, nan=1.0)


def run_views(model, batch, off, gen, device, beta, gamma, keep_prob=0.7, sigma=0.0):
    wa, wb = batch
    Xa, Xb = wa["X"].to(device), wb["X"].to(device)
    va = make_view(Xa, wa["r"].to(device), wa["a"].to(device), off, gen, keep_prob)
    vb = make_view(Xb, wb["r"].to(device), wb["a"].to(device), off, gen, keep_prob)
    return model.losses(va, vb, Xa, Xb, beta=beta, gamma=gamma, sigma=sigma)


def evaluate(model, fold, off, device, beta, gamma, seed: int = 123, keep_prob: float = 0.7, sigma: float = 0.0) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TwoViewDataset(fold.val, seed=seed), batch_size=128, shuffle=False, drop_last=True)
    terms = {}
    with torch.no_grad():
        for batch in loader:
            L = run_views(model, batch, off, gen, device, beta, gamma, keep_prob, sigma)
            for k in ("total", "A", "B", "C", "C_inv", "C_var", "C_cov", "S"):
                terms.setdefault(k, []).append(L[k].item())
    tr, va = embed(model.encoder, fold.train, device), embed(model.encoder, fold.val, device)
    std = va["e"].std(0)
    out = {f"val_{k}": float(np.mean(v)) for k, v in terms.items()}
    out |= {"e_std_min": float(std.min()), "e_std_mean": float(std.mean()),
            "eff_rank_val": effective_rank(va["e"]), "eff_rank_train": effective_rank(tr["e"])}
    out |= {f"probe_{k}": v for k, v in probe_r2(tr, va).items()}
    out |= {f"probe_{k}": v for k, v in evaluate_encoder(model.encoder, fold, device, SELECTION_REGIMES).items()}
    return out


def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out_dir = ROOT / "runs" / args.run_name / f"fold{args.test_pig}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold = make_fold(select_recordings(args.pigs, args.no_flow_pigs), args.test_pig, args.val_pig,
                     vlv_mode=args.vlv_mode, qao_mode=args.qao_mode, per_recording=args.per_recording)
    fold.normalizer.save(out_dir / "normalizer.json")
    print(f"fold: test pig {fold.test_pig}, val pig {fold.val_pig}, train pigs {fold.train_pigs} | "
          f"windows train {len(fold.train)}, val {len(fold.val)}, test {len(fold.test)} | device {device}")

    two = TwoViewDataset(fold.train, seed=args.seed)
    sampler = SpacedBatchSampler(fold.train, args.batch_size, seed=args.seed)
    loader = DataLoader(two, batch_sampler=sampler)
    off = offset_scale(fold.normalizer, device)
    gen = torch.Generator().manual_seed(args.seed)

    model = SSLModel().to(device)
    model.set_summary_stats(*(t.to(device) for t in summary_stats(fold.train)))
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    total = args.epochs * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, 0.05, args.lr, args.lr_min))

    log, best = [], -np.inf
    ev0 = evaluate(model, fold, off, device, args.beta, args.gamma, keep_prob=args.keep_prob, sigma=args.sigma)
    s0 = selection_score(ev0)
    log.append({"epoch": 0} | ev0 | {"probe_mean": s0})
    print(f"ep   0 | untrained | eff.rank val {ev0['eff_rank_val']:.1f} | within-pig R² " +
          " ".join(f"{k} {ev0[f'probe_{k}@within']:+.2f}" for k in PROBE_TARGETS) +
          f" | Q from P_ao {ev0['probe_P_ao_only:mean_Q_ao@within']:+.2f} no-Q {ev0['probe_no_Q_ao:mean_Q_ao@within']:+.2f}"
          f" | score {s0:+.3f}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, sums, n = time.time(), {}, 0
        for batch in loader:
            L = run_views(model, batch, off, gen, device, args.beta, args.gamma, args.keep_prob, args.sigma)
            opt.zero_grad(set_to_none=True)
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            for k in ("total", "A", "B", "C", "C_inv", "C_var", "C_cov", "S"):
                sums[k] = sums.get(k, 0.0) + L[k].item()
            n += 1
        row = {"epoch": epoch, "lr": sched.get_last_lr()[0], "time_s": time.time() - t0, "batches": n}
        row |= {f"train_{k}": v / n for k, v in sums.items()}
        msg = (f"ep {epoch:3d} | {row['time_s']:5.1f}s | train total {row['train_total']:.3f} "
               f"A {row['train_A']:.3f} B {row['train_B']:.3f} C {row['train_C']:.2f} S {row['train_S']:.3f} "
               f"(inv {row['train_C_inv']:.3f} var {row['train_C_var']:.3f} cov {row['train_C_cov']:.3f})")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ev = evaluate(model, fold, off, device, args.beta, args.gamma, keep_prob=args.keep_prob, sigma=args.sigma)
            row |= ev
            # R² clipped at −1: one target that fails badly on the validation pig
            # (e.g. a different flow calibration) must not dominate checkpoint selection
            score = selection_score(ev)
            row["probe_mean"] = float(score)
            msg += (f"\n         val total {ev['val_total']:.3f} A {ev['val_A']:.3f} B {ev['val_B']:.3f} | "
                    f"eff.rank val {ev['eff_rank_val']:.1f} train {ev['eff_rank_train']:.1f} | "
                    f"e std min {ev['e_std_min']:.3f} | within-pig R² " +
                    " ".join(f"{k} {ev[f'probe_{k}@within']:+.2f}" for k in PROBE_TARGETS) +
                    f" | Q from P_ao {ev['probe_P_ao_only:mean_Q_ao@within']:+.2f} no-Q {ev['probe_no_Q_ao:mean_Q_ao@within']:+.2f}"
                    f" | score {score:+.3f}")
            torch.save({"encoder": model.encoder.state_dict(), "epoch": epoch, "score": score,
                        "args": vars(args)}, out_dir / f"encoder_ep{epoch}.pt")
            if score > best:
                best = score
                torch.save({"encoder": model.encoder.state_dict(), "epoch": epoch, "probe_mean": score,
                            "args": vars(args)}, out_dir / "encoder_best.pt")
                msg += "  ← best"
        print(msg, flush=True)
        log.append(row)
        (out_dir / "log.json").write_text(json.dumps(log, indent=1))

    torch.save({"model": model.state_dict(), "epoch": args.epochs, "args": vars(args)}, out_dir / "ssl_last.pt")
    print(f"done. best mean probe R² {best:+.3f}; outputs in {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-pig", type=int, required=True)
    ap.add_argument("--val-pig", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=0.0, help="weight of the summary objective L_S (0 = spec)")
    ap.add_argument("--per-recording", action="store_true",
                    help="normalize each recording with its own statistics (option B2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="default")
    ap.add_argument("--keep-prob", type=float, default=0.7, help="sensor dropout: P(keep a signal) per view")
    ap.add_argument("--pigs", type=int, nargs="+", default=None, help="recordings to use (default: all)")
    ap.add_argument("--no-flow-pigs", type=int, nargs="+", default=None,
                    help="recordings whose Q_ao is treated as unavailable")
    ap.add_argument("--qao-mode", default="scaled", choices=["zero", "scaled"])
    ap.add_argument("--vlv-mode", default="scaled", choices=["absolute", "relative", "scaled"])
    main(ap.parse_args())
