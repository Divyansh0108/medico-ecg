"""Shared helpers for the Track 2 follow-up analyses (RULES2.md): grid loading, 3-seed averages, group
scores and patient bootstraps that reuse one set of resamples for every model."""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "results", "robustness")
SEEDS = [0, 1, 2]
REGIMES = {"B0-clean": ["resnet1d_wang_crop_dsnorm", "resnet1d_wang_crop_dsnorm_s1", "resnet1d_wang_crop_dsnorm_s2"],
           **{k: [f"{p}_s{s}" for s in SEEDS] for k, p in
              [("B0-aug", "B0aug"), ("C", "C"), ("D", "D"), ("E", "E"), ("F", "F"), ("E-clean", "Eclean")]}}
GATED = ["D", "E", "F", "E-clean"]
COLORS = {"B0-clean": "#2a78d6", "B0-aug": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4",
          "F": "#008300", "E-clean": "#4a3aa7", "B0-aug-real": "#9c4a1a", "F-real": "#0b5d0b"}
MARKERS = {"B0-clean": "o", "B0-aug": "s", "C": "^", "D": "D", "E": "v", "F": "P", "E-clean": "X",
           "B0-aug-real": "s", "F-real": "P"}


def fast_macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Macro AUROC via Mann-Whitney ranks (ties averaged); equals sklearn roc_auc_score(average='macro')."""
    r = rankdata(p, axis=0)
    n1 = y.sum(0)
    n0 = len(y) - n1
    return float(np.mean(((r * y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Binary AUROC (ties averaged)."""
    return fast_macro_auroc(np.asarray(y, float)[:, None], np.asarray(s, float)[:, None])


class Grid:
    """One evaluation grid (results/robustness/grid/<name>): per-tag probs (C, N, 5), r (C, N)."""

    def __init__(self, name: str, regimes: dict[str, list[str]] = REGIMES):
        g = os.path.join(D, "grid", name)
        self.regimes = {k: ts for k, ts in regimes.items() if all(os.path.exists(os.path.join(g, f"{t}.npz")) for t in ts)}
        self.Z = {t: np.load(os.path.join(g, f"{t}.npz")) for ts in self.regimes.values() for t in ts}
        z0 = next(iter(self.Z.values()))
        self.names, self.Y, self.PID, self.ids = list(z0["conds"]), z0["y"], z0["patient_id"], z0["ecg_id"]
        for t, z in self.Z.items():
            assert list(z["conds"]) == self.names and np.array_equal(z["ecg_id"], self.ids), t
        self.ix = {c: k for k, c in enumerate(self.names)}
        self.ens = {k: np.mean([self.Z[t]["probs"] for t in ts], 0) for k, ts in self.regimes.items()}
        self.seed = {k: [self.Z[t]["probs"] for t in ts] for k, ts in self.regimes.items()}
        self.rens = {k: np.mean([self.Z[t]["r"] for t in ts], 0) for k, ts in self.regimes.items()
                     if not np.isnan(self.Z[ts[0]]["r"]).all()}

    def score(self, P, conds, idx=None) -> float:
        sl = slice(None) if idx is None else idx
        return float(np.mean([fast_macro_auroc(self.Y[sl], P[self.ix[c]][sl]) for c in conds]))

    def resamples(self, n: int = 1000, seed: int = 0):
        """Patient resamples, same rejection rule and RNG sequence as metrics.paired_patient_bootstrap."""
        rng = np.random.default_rng(seed)
        pids = np.unique(self.PID)
        idx_of = {p: np.flatnonzero(self.PID == p) for p in pids}
        out = []
        while len(out) < n:
            idx = np.concatenate([idx_of[p] for p in rng.choice(pids, len(pids), replace=True)])
            if (self.Y[idx].sum(0) == 0).any() or (self.Y[idx].sum(0) == len(idx)).any():
                continue
            out.append(idx)
        return out

    def boot(self, Ps: dict[str, np.ndarray], groups: dict[str, list[str]], n: int = 1000, seed: int = 0):
        """{group: {model: (n,) group score per resample}} on shared resamples."""
        conds = sorted({c for cs in groups.values() for c in cs})
        res = self.resamples(n, seed)
        per = {m: np.array([[fast_macro_auroc(self.Y[i], P[self.ix[c]][i]) for c in conds] for i in res]) for m, P in Ps.items()}
        ci = {c: k for k, c in enumerate(conds)}
        return {g: {m: per[m][:, [ci[c] for c in cs]].mean(1) for m in Ps} for g, cs in groups.items()}


def paired(bs: dict, full: dict, a: str, b: str) -> dict:
    """Paired difference a - b from shared resamples (bs: model -> (n,)); point estimate on full data."""
    d = bs[a] - bs[b]
    return {"diff": full[a] - full[b], "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975)),
            "p_le_0": float((d <= 0).mean())}


def ci(bs: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def spearman(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def snr_of(c: str) -> float:
    return float(c.split("|")[1])


def f4(v: float) -> str:
    return f"{v:.4f}"


def fmt_boot(b: dict) -> str:
    return f"{b['diff']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]"


def mean_std(vals) -> str:
    return f"{np.mean(vals):.4f} +- {np.std(vals, ddof=1):.4f}"
