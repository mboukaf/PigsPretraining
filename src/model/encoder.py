"""Step 10: the encoder (spec §12). Window of beats → embedding e.

    X (B, W, M, L), r (B, W), a (B, W, M)
      12.1 per-signal beat CNNs            → U (B, W, M, C)
      12.2 missing-signal replacement      → U (B, W, M, C)
      12.3 fusion across signals           → (B, W, D)        one token per beat
      12.4 + position + duration           → (B, W, D)
      12.5 + CLS token                     → (B, 1 + W, D)
      12.6 transformer encoder             → H (B, 1 + W, D)
      12.7 embedding head on CLS           → e (B, d_e)

Hidden beats (training-time beat masking, §13.1) are excluded through the attention
padding mask rather than physically removed from the sequence: for the CLS token and
the visible beats this is mathematically the same, and it keeps the tensors fixed-size.
Beats with no available signal at all (rejected beats) are excluded the same way.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EncoderConfig:
    n_signals: int = 6         # M
    n_points: int = 64         # L
    n_beats: int = 32          # W
    beat_dim: int = 64         # C, output of each per-signal CNN
    d_model: int = 128         # D, token width
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    d_embed: int = 32          # d_e


class BeatCNN(nn.Module):
    """One beat of one signal (L points) → vector of C numbers (§12.1).
    Flatten instead of pooling: the position of features inside the beat is information."""

    def __init__(self, n_points: int = 64, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3), 
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), 
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Flatten(),
        )
        self.proj = nn.Linear(32 * (n_points // 4), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # (n, L) → (n, C)
        return self.proj(self.conv(x.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.cfg = cfg
        M, C, D = cfg.n_signals, cfg.beat_dim, cfg.d_model
        self.beat_cnns = nn.ModuleList(BeatCNN(cfg.n_points, C) for _ in range(M))     # 12.1
        self.missing = nn.Parameter(torch.randn(M, C) * 0.02)                           # 12.2
        self.fusion = nn.Sequential(nn.Linear(M * C, D), nn.LayerNorm(D))               # 12.3
        self.pos = nn.Parameter(torch.randn(cfg.n_beats, D) * 0.02)                     # 12.4
        self.dur = nn.Linear(1, D)
        self.cls = nn.Parameter(torch.randn(D) * 0.02)                                  # 12.5
        layer = nn.TransformerEncoderLayer(D, cfg.n_heads, 
                                           cfg.d_ff,
                                             cfg.dropout,       # 12.6
                                           activation="gelu", 
                                           batch_first=True, 
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, cfg.d_embed))           # 12.7

    def tokens(self, X: torch.Tensor, r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Beat tokens (B, W, D) before the transformer (12.1–12.4)."""
        B, W, M, L = X.shape
        U = torch.stack([cnn(X[:, :, m].reshape(B * W, L)) for m, cnn in enumerate(self.beat_cnns)], dim=1)
        U = U.reshape(B, W, M, -1)
        U = torch.where(a.unsqueeze(-1), U, self.missing.expand_as(U))                 # 12.2
        tok = self.fusion(U.reshape(B, W, -1))                                          # 12.3
        return tok + self.pos[:W] + self.dur(r.unsqueeze(-1))                           # 12.4

    def forward(self, X: torch.Tensor, r: torch.Tensor, a: torch.Tensor,
                visible: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        X (B, W, M, L) float, r (B, W) float, a (B, W, M) bool,
        visible (B, W) bool: beats given to the encoder (default: all).
        Returns e (B, d_e), H (B, 1 + W, D) and the key padding mask used (B, 1 + W).
        """
        B, W = r.shape
        if visible is None:
            visible = torch.ones(B, W, dtype=torch.bool, device=X.device)
        visible = visible & a.any(-1)                                   # no signal → not a token
        seq = torch.cat([self.cls.expand(B, 1, -1), self.tokens(X, r, a)], dim=1)       # 12.5
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=X.device), ~visible], dim=1)
        H = self.transformer(seq, src_key_padding_mask=pad)
        return {"e": self.head(H[:, 0]), "H": H, "pad": pad}


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
