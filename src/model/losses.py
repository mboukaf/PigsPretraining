"""Step 11, part 3: the three training losses and their combination (spec §13.2–13.5).

    L = L_A(a) + L_A(b) + β [L_B(a) + L_B(b)] + γ L_C(a, b) + σ [L_S(a) + L_S(b)]

L_S (summary objective, not in the spec, Docs/experiments.md E11): from e alone, predict the
window's summary statistics (model/features.py) computed on the full clean window, standardized
with statistics of the training set. σ = 0 reproduces the spec's objective.

γ default 0.1: measured at initialization on real batches, L_A ≈ 3.1, L_B ≈ 1.2 and
L_C ≈ 37 (dominated by the variance term), so γ ≈ (L_A + L_B) / L_C ≈ 0.115 balances the
terms. The spec's first guess (0.02–0.05) assumed reconstruction errors below 1.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .encoder import Encoder, EncoderConfig
from .features import window_summaries
from .heads import Projector, ReconstructionDecoder, SummaryHead, TemplateHead

VICREG_WEIGHTS = {"inv": 25.0, "var": 25.0, "cov": 1.0}


def reconstruction_loss(pred: torch.Tensor, X: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(A) MSE between reconstruction and clean beats, only on `target` entries (B, W, M)."""
    w = target.unsqueeze(-1).float()
    return ((pred - X) ** 2 * w).sum() / (w.sum() * X.shape[-1]).clamp(min=1)


def window_templates(X: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean beat and per-point std across the present beats of each window and signal.
    X (B, W, M, L), present (B, W, M) → template (B, 2, M, L), signal_ok (B, M)."""
    w = present.unsqueeze(-1).float()
    n = w.sum(dim=1)                                                     # (B, M, 1)
    mean = (X * w).sum(dim=1) / n.clamp(min=1)
    var = (((X - mean.unsqueeze(1)) ** 2) * w).sum(dim=1) / (n - 1).clamp(min=1)
    return torch.stack([mean, var.sqrt()], dim=1), n.squeeze(-1) >= 2


def template_loss(pred: torch.Tensor, X: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """(B) MSE of the predicted mean / std beats, over signals present in the window."""
    target, ok = window_templates(X, present)
    w = ok[:, None, :, None].float()                                     # (B, 1, M, 1)
    return ((pred - target) ** 2 * w).sum() / (w.sum() * 2 * X.shape[-1]).clamp(min=1)


def vicreg_loss(za: torch.Tensor, zb: torch.Tensor, eps: float = 1e-4) -> dict[str, torch.Tensor]:
    """(C) VICReg: invariance + variance (std ≥ 1 per dimension) + covariance (decorrelation)."""
    B, D = za.shape
    inv = F.mse_loss(za, zb)
    var = cov = 0.0
    for z in (za, zb):
        z = z - z.mean(dim=0)
        var = var + F.relu(1 - torch.sqrt(z.var(dim=0) + eps)).mean()
        c = (z.T @ z) / (B - 1)
        cov = cov + (c - torch.diag(torch.diag(c))).pow(2).sum() / D
    w = VICREG_WEIGHTS
    return {"inv": inv, "var": var, "cov": cov, "total": w["inv"] * inv + w["var"] * var + w["cov"] * cov}


def summary_loss(pred: torch.Tensor, X: torch.Tensor, r: torch.Tensor, present: torch.Tensor,
                 mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """(S) MSE between predicted and actual standardized window summaries, over present entries."""
    target = (window_summaries(X, r, present) - mean) / std
    ok = ~torch.isnan(target)
    return ((pred - torch.nan_to_num(target)) ** 2 * ok).sum() / ok.sum().clamp(min=1)


class SSLModel(nn.Module):
    """Encoder + the three training-only heads."""

    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = ReconstructionDecoder(cfg)
        self.template = TemplateHead(cfg)
        self.projector = Projector(cfg.d_embed)
        self.summary = SummaryHead(cfg.d_embed, 6 * cfg.n_signals + 2)
        # standardization of the summary targets (set from the training set by set_summary_stats)
        self.register_buffer("sum_mean", torch.zeros(6 * cfg.n_signals + 2))
        self.register_buffer("sum_std", torch.ones(6 * cfg.n_signals + 2))

    def set_summary_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.sum_mean.copy_(mean)
        self.sum_std.copy_(std.clamp(min=1e-3))

    def view_outputs(self, v: dict) -> dict[str, torch.Tensor]:
        enc = self.encoder(v["X_in"], v["r"], v["a_in"], v["visible"])
        return {"e": enc["e"],
                "recon": self.decoder(enc["H"], enc["pad"], enc["e"], v["r"]),
                "template": self.template(enc["e"]),
                "z": self.projector(enc["e"]),
                "summary": self.summary(enc["e"])}

    def losses(self, va: dict, vb: dict, Xa: torch.Tensor, Xb: torch.Tensor,
               beta: float = 1.0, gamma: float = 0.1, sigma: float = 0.0) -> dict[str, torch.Tensor]:
        """va / vb: views from make_view; Xa / Xb: the clean normalized windows they came from."""
        oa, ob = self.view_outputs(va), self.view_outputs(vb)
        L_A = reconstruction_loss(oa["recon"], Xa, va["target"]) + reconstruction_loss(ob["recon"], Xb, vb["target"])
        L_B = template_loss(oa["template"], Xa, va["present"]) + template_loss(ob["template"], Xb, vb["present"])
        C = vicreg_loss(oa["z"], ob["z"])
        L_S = (summary_loss(oa["summary"], Xa, va["r"], va["present"], self.sum_mean, self.sum_std)
               + summary_loss(ob["summary"], Xb, vb["r"], vb["present"], self.sum_mean, self.sum_std))
        total = L_A + beta * L_B + gamma * C["total"] + sigma * L_S
        return {"total": total, "A": L_A, "B": L_B, "C": C["total"], "S": L_S,
                "C_inv": C["inv"], "C_var": C["var"], "C_cov": C["cov"], "e_a": oa["e"], "e_b": ob["e"]}
