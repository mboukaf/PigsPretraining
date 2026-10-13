"""Step 11, part 2: training-only components (spec §13.2–13.4). Discarded after training.

(A) ReconstructionDecoder: rebuilds the beats the encoder did not see (hidden beats and
    dropped signals) from the encoder's beat outputs and the embedding e.
(B) TemplateHead: from e alone, the window's mean beat and per-point std across beats,
    for every signal (targets from the full window, before dropout and masking).
(C) Projector: e → z for the VICReg agreement loss between the two views.
"""
import torch
from torch import nn

from .encoder import EncoderConfig


class ReconstructionDecoder(nn.Module):
    """Input (B, 1 + W, D_dec): [e token] + one token per beat, where beats the encoder did
    not attend get a learned mask vector. Output (B, W, M, L)."""

    def __init__(self, cfg: EncoderConfig = EncoderConfig(), d_dec: int = 64, n_layers: int = 2,
                 n_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.d_model, d_dec)
        self.mask_token = nn.Parameter(torch.randn(d_dec) * 0.02)
        self.pos = nn.Parameter(torch.randn(cfg.n_beats, d_dec) * 0.02)
        self.dur = nn.Linear(1, d_dec)
        self.e_proj = nn.Linear(cfg.d_embed, d_dec)
        layer = nn.TransformerEncoderLayer(d_dec, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_dec)
        self.out = nn.Linear(d_dec, cfg.n_signals * cfg.n_points)

    def forward(self, H: torch.Tensor, pad: torch.Tensor, e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """H (B, 1+W, D) and pad (B, 1+W) from the encoder, e (B, d_e), r (B, W)."""
        B, W = r.shape
        beats = self.proj(H[:, 1:])
        beats = torch.where(pad[:, 1:, None], self.mask_token.expand_as(beats), beats)
        beats = beats + self.pos[:W] + self.dur(r.unsqueeze(-1))
        seq = torch.cat([self.e_proj(e).unsqueeze(1), beats], dim=1)
        out = self.out(self.norm(self.transformer(seq))[:, 1:])
        return out.view(B, W, self.cfg.n_signals, self.cfg.n_points)


class TemplateHead(nn.Module):
    """e (B, d_e) → (B, 2, M, L): [mean beat, std across beats] per signal."""

    def __init__(self, cfg: EncoderConfig = EncoderConfig(), hidden: int = 256):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(nn.Linear(cfg.d_embed, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * cfg.n_signals * cfg.n_points))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e).view(-1, 2, self.cfg.n_signals, self.cfg.n_points)


class Projector(nn.Module):
    """e (B, d_e) → z (B, 128) for VICReg."""

    def __init__(self, d_embed: int = 32, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


class SummaryHead(nn.Module):
    """e (B, d_e) → window summary statistics (B, n_out), see model/features.py (objective S)."""

    def __init__(self, d_embed: int = 32, n_out: int = 38, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_embed, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)
