from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from loaders.ptbxl import SUPERCLASSES


def per_class_auroc(y: np.ndarray, p: np.ndarray, classes=SUPERCLASSES) -> dict[str, float]:
    return {c: float(roc_auc_score(y[:, i], p[:, i])) for i, c in enumerate(classes)}


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p, average="macro"))


def patient_bootstrap_ci(y, p, patient_id, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Resample patients with replacement; all records of a drawn patient are included."""
    rng = np.random.default_rng(seed)
    pids = np.unique(patient_id)
    idx_of = {pid: np.flatnonzero(patient_id == pid) for pid in pids}
    stats = []
    while len(stats) < n:
        draw = rng.choice(pids, size=len(pids), replace=True)
        idx = np.concatenate([idx_of[d] for d in draw])
        if (y[idx].sum(0) == 0).any() or (y[idx].sum(0) == len(idx)).any():
            continue
        stats.append(macro_auroc(y[idx], p[idx]))
    stats = np.asarray(stats)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)), stats


def paired_patient_bootstrap(y, pa, pb, patient_id, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Macro-AUROC(a) - macro-AUROC(b) on the same resampled patients. Returns (diff, lo, hi, P(diff <= 0))."""
    rng = np.random.default_rng(seed)
    pids = np.unique(patient_id)
    idx_of = {pid: np.flatnonzero(patient_id == pid) for pid in pids}
    d = []
    while len(d) < n:
        idx = np.concatenate([idx_of[k] for k in rng.choice(pids, size=len(pids), replace=True)])
        if (y[idx].sum(0) == 0).any() or (y[idx].sum(0) == len(idx)).any():
            continue
        d.append(macro_auroc(y[idx], pa[idx]) - macro_auroc(y[idx], pb[idx]))
    d = np.asarray(d)
    return (macro_auroc(y, pa) - macro_auroc(y, pb), float(np.quantile(d, alpha / 2)),
            float(np.quantile(d, 1 - alpha / 2)), float((d <= 0).mean()))


def ece_binary(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
