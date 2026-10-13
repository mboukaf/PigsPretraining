"""Step 12 monitoring (spec §14.2): embeddings of clean windows, collapse indicators and a
quick linear probe on the validation pig.

Probe targets are computed per window from the normalized clean data. They are affine
transforms of physical quantities (mean pressure, mean flow, ...), which leaves R² unchanged:
    heart_period : mean normalized log beat duration (r)
    mean_P_ao    : mean aortic pressure
    pp_P_ao      : mean pulse pressure of P_ao (max − min of each beat)
    mean_Q_ao    : mean aortic flow (∝ cardiac output, zero referenced)

Two scores per target, both on the validation pig:
    R²          absolute: the probe must also get the pig's average level right
    R²@within   predictions and targets centred on the validation pig first: does the probe
                follow the changes within the new animal? Constant between-pig offsets (flow
                calibration, waveform shape after per-recording scaling) do not count.
See Docs/experiments.md (E7).
"""
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from preprocessing.config import CARDIAC

from data.windows import WindowDataset

PROBE_TARGETS = ("heart_period", "mean_P_ao", "pp_P_ao", "mean_Q_ao")
_AO, _Q = CARDIAC.index("P_ao"), CARDIAC.index("Q_ao")


@torch.no_grad()
def embed(encoder, ds: WindowDataset, device, batch_size: int = 256) -> dict[str, np.ndarray]:
    """Embeddings of all windows of a dataset (no views, no masking) + probe targets."""
    encoder.eval()
    E, T = [], {k: [] for k in PROBE_TARGETS}
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False):
        X, r, a = b["X"], b["r"], b["a"]
        E.append(encoder(X.to(device), r.to(device), a.to(device))["e"].cpu())
        ao, q = a[:, :, _AO], a[:, :, _Q]
        n_ao, n_q = ao.sum(1).clamp(min=1), q.sum(1).clamp(min=1)
        beat_mean_ao = X[:, :, _AO].mean(-1)
        beat_pp_ao = X[:, :, _AO].amax(-1) - X[:, :, _AO].amin(-1)
        T["heart_period"].append(r.mean(1))
        T["mean_P_ao"].append(torch.where(ao.any(1), (beat_mean_ao * ao).sum(1) / n_ao, torch.nan))
        T["pp_P_ao"].append(torch.where(ao.any(1), (beat_pp_ao * ao).sum(1) / n_ao, torch.nan))
        T["mean_Q_ao"].append(torch.where(q.any(1), (X[:, :, _Q].mean(-1) * q).sum(1) / n_q, torch.nan))
    return {"e": torch.cat(E).numpy()} | {k: torch.cat(v).numpy() for k, v in T.items()}


def effective_rank(e: np.ndarray) -> float:
    """exp(entropy of the normalized covariance eigenvalues): ≈ d_e if all dims used, ≈ 1 if collapsed."""
    lam = np.clip(np.linalg.eigvalsh(np.cov(e.T)), 0, None)
    p = lam / lam.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def probe_r2(train: dict, val: dict, max_train: int = 5000, seed: int = 0) -> dict[str, float]:
    """Ridge regression e → target, fitted on training-pig windows, scored on the validation pig.
    Returns {target: R², target@within: within-pig R²}."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in PROBE_TARGETS:
        tr, va = ~np.isnan(train[k]), ~np.isnan(val[k])
        if tr.sum() < 50 or va.sum() < 20:
            out[k] = out[f"{k}@within"] = float("nan")
            continue
        idx = np.flatnonzero(tr)
        idx = rng.choice(idx, min(max_train, len(idx)), replace=False)
        sc = StandardScaler().fit(train["e"][idx])
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(train["e"][idx]), train[k][idx])
        y, yhat = val[k][va], model.predict(sc.transform(val["e"][va]))
        out[k] = float(r2_score(y, yhat))
        out[f"{k}@within"] = float(r2_score(y - y.mean(), yhat - yhat.mean()))
    return out
