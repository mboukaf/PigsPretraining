"""Window summary statistics, shared by the benchmark's handcrafted baseline and the summary
training objective (Docs/experiments.md, E11).

For each signal, over the window's beats where it is available: mean, min, max, peak-to-peak and
maximum slope of the beat (averaged over beats), and the beat-to-beat std of the peak-to-peak;
plus the mean and std of the normalized beat durations r. Absent signals give NaN.
"""
import torch

N_STATS = 6


def window_summaries(X: torch.Tensor, r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """X (B, W, M, L), r (B, W), a (B, W, M) → (B, N_STATS·M + 2), NaN where a signal is absent.
    Column order: [stat 0 for all M signals, stat 1 for all M, …, mean r, std r]."""
    w = a.float()
    n = w.sum(1)                                                            # (B, M)

    def avg(v):                                                            # (B, W, M) → (B, M)
        return torch.where(n > 0, (v * w).sum(1) / n.clamp(min=1), torch.full_like(n, float("nan")))

    ptp = X.amax(-1) - X.amin(-1)
    slope = (X[..., 1:] - X[..., :-1]).amax(-1)
    ptp_mean = avg(ptp)
    ptp_var = (((ptp - torch.nan_to_num(ptp_mean)[:, None]) ** 2) * w).sum(1) / (n - 1).clamp(min=1)
    ptp_std = torch.where(n > 1, ptp_var.sqrt(), torch.full_like(n, float("nan")))
    stats = [avg(X.mean(-1)), avg(X.amin(-1)), avg(X.amax(-1)), ptp_mean, avg(slope), ptp_std]
    return torch.cat(stats + [r.mean(1, keepdim=True), r.std(1, keepdim=True)], dim=1)
