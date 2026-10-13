"""Evaluation targets per window (spec §15.1). Never used for training.

Per-beat quantities are computed from the referenced beats (data.normalize.reference with the
scaled modes): pressures in mmHg, Q_ao and V_lv in units relative to the recording's typical
beat amplitude (their absolute calibration is not comparable between pigs). Window targets
are averages over the window's beats where the source signals are available, plus a few
window statistics (pulse-pressure variation, PEEP).

Groups (source signals in brackets):
    timing     heart_rate [rr], ptt [P_ao, P_fem]
    pressure   map_ao, sbp_ao, dbp_ao, pp_ao, ppv_ao [P_ao]; map_fem, pp_fem [P_fem]
    flow       sv_rel, co_rel [Q_ao]
    load       wk_R, wk_tau, wk_C, wk_Zc [P_ao, Q_ao]  (2-element Windkessel + characteristic impedance)
    lv         edv_rel, sv_lv_rel [V_lv]; dpdt_max_lv, edp_lv [P_lv]
    venous     map_vc [P_vc]
    context    peep [airway pressure — never an encoder input]
"""
import h5py
import numpy as np
import pandas as pd

from preprocessing.config import CARDIAC, FS, SIGNALS

from data.normalize import reference
from data.store import PIGS_H5, Recording
from data.windows import W, window_availability

L = 64
_i = {s: CARDIAC.index(s) for s in CARDIAC}

TARGET_SOURCES = {
    "heart_rate": [], "ptt": ["P_ao", "P_fem"],
    "map_ao": ["P_ao"], "sbp_ao": ["P_ao"], "dbp_ao": ["P_ao"], "pp_ao": ["P_ao"], "ppv_ao": ["P_ao"],
    "map_fem": ["P_fem"], "pp_fem": ["P_fem"],
    "sv_rel": ["Q_ao"], "co_rel": ["Q_ao"],
    "wk_R": ["P_ao", "Q_ao"], "wk_tau": ["P_ao"], "wk_C": ["P_ao", "Q_ao"], "wk_Zc": ["P_ao", "Q_ao"],
    "edv_rel": ["V_lv"], "sv_lv_rel": ["V_lv"], "dpdt_max_lv": ["P_lv"], "edp_lv": ["P_lv"],
    "map_vc": ["P_vc"], "peep": [],
}
TARGET_GROUPS = {
    "timing": ["heart_rate", "ptt"],
    "pressure": ["map_ao", "sbp_ao", "dbp_ao", "pp_ao", "ppv_ao", "map_fem", "pp_fem"],
    "flow": ["sv_rel", "co_rel"],
    "load": ["wk_R", "wk_tau", "wk_C", "wk_Zc"],
    "lv": ["edv_rel", "sv_lv_rel", "dpdt_max_lv", "edp_lv"],
    "venous": ["map_vc"],
    "context": ["peep"],
}
TARGETS = [t for g in TARGET_GROUPS.values() for t in g]


def beat_airway_min(pig: int, boundaries: np.ndarray) -> np.ndarray:
    """Minimum airway pressure within each beat (≈ PEEP for beats in expiration)."""
    with h5py.File(PIGS_H5, "r") as f:
        paw = f[f"pig{pig}/filtered"][SIGNALS.index("P_aw")]
    x = np.where(np.isnan(paw), np.inf, paw)
    m = np.minimum.reduceat(x, boundaries[:-1])
    return np.where(np.isfinite(m), m, np.nan)


def beat_quantities(rec: Recording) -> dict[str, np.ndarray]:
    """Per-beat quantities (K,), NaN where a source signal is unavailable."""
    X = reference(rec, vlv_mode="scaled", qao_mode="scaled").X.astype(np.float64)
    av, rr = rec.avail, rec.rr.astype(np.float64)
    dt = rr[:, None] / L                                                     # (K, 1) s per point
    nan = lambda ok, v: np.where(ok, v, np.nan)
    ao, q, fem, lv, vlv, vc = (X[:, _i[s]] for s in CARDIAC)
    a = {s: av[:, _i[s]] for s in CARDIAC}

    out = {"heart_rate": 60.0 / rr}
    up_ao = np.argmax(np.diff(ao, axis=1), axis=1)
    up_fem = np.argmax(np.diff(fem, axis=1), axis=1)
    out["ptt"] = nan(a["P_ao"] & a["P_fem"], (up_fem - up_ao) * dt[:, 0])
    out |= {"map_ao": nan(a["P_ao"], ao.mean(1)), "sbp_ao": nan(a["P_ao"], ao.max(1)),
            "dbp_ao": nan(a["P_ao"], ao.min(1)), "pp_ao": nan(a["P_ao"], np.ptp(ao, axis=1)),
            "map_fem": nan(a["P_fem"], fem.mean(1)), "pp_fem": nan(a["P_fem"], np.ptp(fem, axis=1))}

    sv = q.sum(1) * dt[:, 0]                                                 # relative flow × s
    out |= {"sv_rel": nan(a["Q_ao"], sv), "co_rel": nan(a["Q_ao"], sv * 60.0 / rr)}

    # Windkessel: R = mean P / mean Q; τ from the diastolic decay of P_ao (last 40 % of the beat,
    # log-linear fit); C = τ / R; Zc = early systolic pressure rise / peak flow
    pq = a["P_ao"] & a["Q_ao"]
    qmean = q.mean(1)
    out["wk_R"] = nan(pq & (qmean > 1e-3), ao.mean(1) / np.where(qmean > 1e-3, qmean, np.nan))
    tail = slice(int(0.6 * L), L)
    t = np.arange(L)[tail][None, :] * dt
    logp = np.log(np.clip(ao[:, tail], 1.0, None))
    tc, lc = t - t.mean(1, keepdims=True), logp - logp.mean(1, keepdims=True)
    slope = (tc * lc).sum(1) / (tc ** 2).sum(1)
    tau = np.where(slope < -1e-3, -1.0 / np.where(slope < -1e-3, slope, -1.0), np.nan)
    out["wk_tau"] = nan(a["P_ao"] & (tau < 10), tau)
    out["wk_C"] = out["wk_tau"] / out["wk_R"]
    qpk = np.argmax(q, axis=1)
    rise = ao[np.arange(len(ao)), qpk] - ao.min(1)
    out["wk_Zc"] = nan(pq, rise / np.where(q.max(1) > 1e-3, q.max(1), np.nan))

    out |= {"edv_rel": nan(a["V_lv"], vlv.max(1)), "sv_lv_rel": nan(a["V_lv"], np.ptp(vlv, axis=1)),
            "dpdt_max_lv": nan(a["P_lv"], np.diff(lv, axis=1).max(1) / dt[:, 0]),
            "edp_lv": nan(a["P_lv"], lv[:, 2]),
            "map_vc": nan(a["P_vc"], vc.mean(1))}
    return out


def window_targets(recs: dict[int, Recording], index: list[tuple[int, int]], w: int = W) -> pd.DataFrame:
    """One row per window (pig, start); NaN where the source signals are not present in the window."""
    cache = {}
    rows = []
    for pig, start in index:
        if pig not in cache:
            rec = recs[pig]
            b = np.concatenate([np.round(rec.t_beat * FS).astype(np.int64),
                                [int(round((rec.t_beat[-1] + rec.rr[-1]) * FS))]])
            cache[pig] = (beat_quantities(rec), beat_airway_min(pig, b))
        bq, paw_min = cache[pig]
        rec = recs[pig]
        sl = slice(start, start + w)
        a = window_availability(rec.avail[sl], rec.beat_ok[sl])               # (W, M)
        ok_beat = rec.beat_ok[sl]
        row = {"pig": pig, "start": start}
        for name, srcs in TARGET_SOURCES.items():
            if name in ("ppv_ao", "peep"):
                continue
            m = ok_beat.copy()
            for s in srcs:
                m &= a[:, _i[s]]
            v = bq[name][sl][m]
            row[name] = np.nanmean(v) if m.sum() >= 0.5 * w and np.isfinite(v).any() else np.nan
        pp = bq["pp_ao"][sl][ok_beat & a[:, _i["P_ao"]]]
        row["ppv_ao"] = (np.nanmax(pp) - np.nanmin(pp)) / np.nanmean(pp) if len(pp) >= 0.5 * w else np.nan
        row["peep"] = np.nanpercentile(paw_min[sl][ok_beat], 10) if ok_beat.sum() >= 0.5 * w else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
