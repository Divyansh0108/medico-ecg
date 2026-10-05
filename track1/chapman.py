"""Chapman-Shaoxing (PhysioNet/CinC 2021 copy) for RULES4.md setting S4; same API as data.py.

Four rhythm classes of Zheng et al. 2020 from the SNOMED codes in each header (# Dx:). Records mapping to
no class or to several, and records with a non-finite sample, a length other than 5000 or a flat lead,
are dropped. 500 Hz -> 100 Hz (resample_poly 1/5, 1000 samples), in mV. One record per patient
(patient_id = ecg_id = JS number). Split stratified by class 80/10/10, seed 20261004.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from tqdm import tqdm

from data import CACHE, bandpass

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "chapman")
CLASSES = ["SB", "AFIB", "GSVT", "SR"]
CODES = {"SB": {"426177001"}, "AFIB": {"164889003", "164890007"},
         "GSVT": {"427084000", "426761007", "713422000", "233896004", "233897008"},
         "SR": {"426783006", "427393009"}}
SPLITS = ["train", "val", "test"]
SPLIT_SEED = 20261004


def label_of(dx: list[str]) -> int | None:
    """Class index if the codes map to exactly one class, else None."""
    hit = {k for k, cs in CODES.items() if cs & set(dx)}
    return CLASSES.index(hit.pop()) if len(hit) == 1 else None


def _read(hea: str) -> tuple[np.ndarray | None, list[str]]:
    lines = open(hea).read().splitlines()
    n_sig, n = int(lines[0].split()[1]), int(lines[0].split()[3])
    gains = [float(lines[1 + i].split()[2].split("/")[0].split("(")[0]) for i in range(n_sig)]
    dx = next((l.split(":", 1)[1].strip().split(",") for l in lines if l.startswith("# Dx:")), [])
    if n_sig != 12 or n != 5000:
        return None, dx
    x = loadmat(hea[:-4] + ".mat")["val"].astype(np.float64) / np.array(gains)[:, None]
    if x.shape != (12, 5000) or not np.all(np.isfinite(x)) or (x.std(1) < 1e-6).any():
        return None, dx
    return x, dx


def _all() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X (N, 12, 1000) mV at 100 Hz, class index y (N,), ids (N,) for every kept record (cached)."""
    cp = os.path.join(CACHE, "chapman_raw.npz")
    if os.path.exists(cp):
        z = np.load(cp)
        return z["X"], z["y"], z["ids"]
    heas = sorted(glob.glob(os.path.join(ROOT, "g*", "JS*.hea")))
    assert len(heas) > 10000, f"Chapman incomplete: {len(heas)} headers"
    X, y, ids, drop = [], [], [], {"no_class": 0, "bad_signal": 0}
    for h in tqdm(heas, desc="chapman"):
        x, dx = _read(h)
        k = label_of(dx)
        if k is None:
            drop["no_class"] += 1
            continue
        if x is None:
            drop["bad_signal"] += 1
            continue
        X.append(resample_poly(x, 1, 5, axis=-1).astype(np.float32))
        y.append(k)
        ids.append(int(os.path.basename(h)[2:-4]))
    X, y, ids = np.stack(X), np.array(y), np.array(ids)
    print(f"chapman: kept {len(y)} of {len(heas)}, dropped {drop}, class counts {np.bincount(y).tolist()}")
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cp, X=X, y=y, ids=ids)
    return X, y, ids


def split_index(y: np.ndarray, ids: np.ndarray) -> dict[str, np.ndarray]:
    """Stratified 80/10/10 split by class (ids sorted within class, then permuted with SPLIT_SEED)."""
    rng = np.random.default_rng(SPLIT_SEED)
    out = {s: [] for s in SPLITS}
    for k in range(len(CLASSES)):
        idx = np.flatnonzero(y == k)
        idx = idx[np.argsort(ids[idx])][rng.permutation(len(idx))]
        a, b = int(round(0.8 * len(idx))), int(round(0.9 * len(idx)))
        out["train"].append(idx[:a]), out["val"].append(idx[a:b]), out["test"].append(idx[b:])
    return {s: np.sort(np.concatenate(v)) for s, v in out.items()}


def load_raw_split(split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X, y, ids = _all()
    i = split_index(y, ids)[split]
    Y = np.eye(len(CLASSES), dtype=np.float32)[y[i]]
    return X[i], Y, pd.DataFrame({"ecg_id": ids[i], "patient_id": ids[i]})


def load_filtered(leads=None) -> tuple[dict, float, float]:
    out = {}
    for s in SPLITS:
        X, Y, meta = load_raw_split(s)
        out[s] = (bandpass(X if leads is None else X[:, leads]), Y, meta)
    return out, float(out["train"][0].mean()), float(out["train"][0].std())


def load_variant(use_bandpass: bool = True, norm: str = "dataset", leads=None) -> dict:
    """Band-passed, dataset-standardized splits (the only variant RULES4.md uses)."""
    assert use_bandpass and norm == "dataset"
    d, mu, sd = load_filtered(leads)
    return {s: (((X - mu) / sd).astype(np.float32), Y, meta) for s, (X, Y, meta) in d.items()}
