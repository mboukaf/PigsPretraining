"""Step 11, part 1: building the two views of each window (spec §13.1).

Time shift (view b only) needs neighbouring beats, so it is done by TwoViewDataset.
Everything else is applied to a batch on the training device by make_view, independently
for each view:
  - sensor dropout: each present signal kept with KEEP_PROB, at least one kept;
  - beat masking: hidden spans of 3–4 beats alternating with visible gaps of 1–6 beats
    (≈ 50 % hidden on average; spans never merge into longer gaps);
  - augmentations on the encoder input only (targets stay clean):
      additive Gaussian noise, NOISE_STD × the signal's normalized scale (= 1);
      constant offset per window on pressures, uniform ± OFFSET_MMHG[signal] mmHg
      (spec ±1 mmHg; ±0.3 for P_vc, whose whole spread is only ≈ 3 mmHg).
Not implemented yet: beat-grid jitter (needs re-resampling from the continuous signals).

Bookkeeping per view (B, W, M):
    present = entry exists in the data (before any dropout)
    input   = entry given to the encoder (signal kept AND beat visible)
    target  = present AND NOT input   → reconstructed by objective A
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.config import CARDIAC

from data.windows import WindowDataset

MAX_SHIFT = 4
KEEP_PROB = 0.7
SPAN_LENGTHS = (3, 4)
GAP_LENGTHS = (1, 2, 3, 4, 5, 6)
NOISE_STD = 0.015
OFFSET_MMHG = {"P_ao": 1.0, "P_fem": 1.0, "P_lv": 1.0, "P_vc": 0.3}


class TwoViewDataset(Dataset):
    """Item = (window at start, window at start + δ), δ uniform in {0..MAX_SHIFT};
    if the shifted window is not valid, the largest valid smaller δ is used."""

    def __init__(self, windows: WindowDataset, max_shift: int = MAX_SHIFT, seed: int = 0):
        self.ds, self.max_shift = windows, max_shift
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        pig, start = self.ds.index[i]
        valid = self.ds.valid_starts[pig]
        delta = int(self.rng.integers(0, self.max_shift + 1))
        while delta > 0 and (start + delta >= len(valid) or not valid[start + delta]):
            delta -= 1
        b = self.ds.window(pig, start + delta)
        b["delta"] = delta
        return self.ds[i], b


def _span_mask(B: int, W: int, gen: torch.Generator, device) -> torch.Tensor:
    """(B, W) bool, True = hidden: alternating visible gaps and hidden spans, random phase."""
    hidden = torch.zeros(B, W, dtype=torch.bool)
    pick = lambda opts: opts[int(torch.randint(len(opts), (1,), generator=gen))]
    for b in range(B):
        pos = int(torch.randint(0, max(GAP_LENGTHS) + 1, (1,), generator=gen))   # random phase
        while pos < W:
            n = pick(SPAN_LENGTHS)
            hidden[b, pos:pos + n] = True
            pos += n + pick(GAP_LENGTHS)
    return hidden.to(device)


def make_view(X: torch.Tensor, r: torch.Tensor, a: torch.Tensor, offset_scale: torch.Tensor,
              gen: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """
    X (B, W, M, L) normalized beats, r (B, W), a (B, W, M) availability (= present),
    offset_scale (M,): OFFSET_MMHG expressed in normalized units per signal (0 for non-pressures).
    Returns the encoder inputs (X_in, r, a_in, visible) and the masks present / target.
    """
    B, W, M, L = X.shape
    dev = X.device
    gen = gen or torch.Generator().manual_seed(int(torch.randint(2**31, (1,))))

    # sensor dropout: keep each present signal with KEEP_PROB, at least one
    sig_present = a.any(dim=1)                                                   # (B, M)
    keep = (torch.rand(B, M, generator=gen) < KEEP_PROB).to(dev) & sig_present
    none = ~keep.any(dim=1)
    if none.any():                                   # force one present signal back in
        scores = torch.rand(B, M, generator=gen).to(dev) * sig_present
        keep[none, scores[none].argmax(dim=1)] = True
    a_in = a & keep[:, None, :]

    # beat masking in spans; never hide every beat that still carries a signal
    hidden = _span_mask(B, W, gen, dev)
    has_signal = a_in.any(dim=2)
    empty = ~(has_signal & ~hidden).any(dim=1)
    hidden[empty] = False
    visible = ~hidden

    # augmentations on the encoder input only
    noise = torch.randn(X.shape, generator=gen).to(dev) * NOISE_STD
    offset = (torch.rand(B, 1, M, 1, generator=gen).to(dev) * 2 - 1) * offset_scale.view(1, 1, M, 1)
    X_in = (X + noise + offset) * (a_in & visible[:, :, None])[..., None]

    inp = a_in & visible[:, :, None]
    return {"X_in": X_in, "r": r, "a_in": a_in, "visible": visible,
            "present": a, "input": inp, "target": a & ~inp, "keep": keep}


def offset_scale(normalizer, device="cpu") -> torch.Tensor:
    """± OFFSET_MMHG on pressures, converted to each signal's normalized units."""
    return torch.tensor([OFFSET_MMHG.get(s, 0.0) / normalizer.scale[s] for s in CARDIAC],
                        dtype=torch.float32, device=device)
