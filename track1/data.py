"""PTB-XL Track 1 data: 100 Hz, 12 leads, 5 diagnostic superclasses (multilabel).

Labels: every scp_codes key with diagnostic == 1 in scp_statements.csv is mapped to
its diagnostic_class (likelihood ignored, as in the official example / Strodthoff et al.).
Records with no superclass are dropped (as in Strodthoff et al. 2021).
Split: strat_fold 1-8 train, 9 val, 10 test.
Preprocessing: band-pass 0.5-40 Hz (4th-order Butterworth, zero-phase), per-record,
per-lead z-score. Output shape per record: (12, 1000).
"""
from __future__ import annotations

import ast
import os
import random

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

ROOT = os.environ.get("PTBXL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ptb-xl"))
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
FS = 100
N_SAMPLES = 1000
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SPLITS = {"train": list(range(1, 9)), "val": [9], "test": [10]}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_meta(root: str = ROOT) -> tuple[pd.DataFrame, np.ndarray]:
    """All records (sorted by ecg_id) with multi-hot superclass labels, BEFORE dropping."""
    db = pd.read_csv(os.path.join(root, "ptbxl_database.csv")).sort_values("ecg_id").reset_index(drop=True)
    stm = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    code2cls = stm.loc[stm["diagnostic"] == 1, "diagnostic_class"].to_dict()
    Y = np.zeros((len(db), len(SUPERCLASSES)), dtype=np.float32)
    for i, codes in enumerate(db["scp_codes"]):
        for code in ast.literal_eval(codes):
            cls = code2cls.get(code)
            if cls in SUPERCLASSES:
                Y[i, SUPERCLASSES.index(cls)] = 1.0
    return db, Y


def bandpass(x: np.ndarray) -> np.ndarray:
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


def zscore_record(y: np.ndarray) -> np.ndarray:
    return (y - y.mean(-1, keepdims=True)) / (y.std(-1, keepdims=True) + 1e-6)


def preprocess(x: np.ndarray) -> np.ndarray:
    """x: (..., 12, 1000) raw mV -> band-passed, per-record per-lead z-scored float32."""
    return zscore_record(bandpass(x)).astype(np.float32)


def _read(path: str) -> np.ndarray:
    rec = wfdb.rdrecord(path, physical=True)
    if rec.fs != FS or rec.sig_len != N_SAMPLES or rec.p_signal.shape[1] != 12:
        raise ValueError(f"{path}: fs={rec.fs} len={rec.sig_len} ch={rec.p_signal.shape}")
    x = rec.p_signal.T
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: non-finite samples")
    return x.astype(np.float32)


def load_split(split: str, root: str = ROOT) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Preprocessed X (N, 12, 1000), Y (N, 5), meta (ecg_id, patient_id, strat_fold) for a split."""
    db, Y = load_meta(root)
    keep = db["strat_fold"].isin(SPLITS[split]).to_numpy() & (Y.sum(1) > 0)
    meta = db.loc[keep, ["ecg_id", "patient_id", "strat_fold", "filename_lr"]].reset_index(drop=True)
    Y = Y[keep]
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f"{split}.npz")
    if os.path.exists(cp):
        z = np.load(cp)
        if np.array_equal(z["ecg_id"], meta["ecg_id"].to_numpy()):
            return z["X"], Y, meta
    X = np.stack([_read(os.path.join(root, f)) for f in tqdm(meta["filename_lr"], desc=f"load {split}")])
    X = preprocess(X)
    np.savez(cp, X=X, ecg_id=meta["ecg_id"].to_numpy())
    return X, Y, meta


def _select(split: str, root: str):
    db, Y = load_meta(root)
    keep = db["strat_fold"].isin(SPLITS[split]).to_numpy() & (Y.sum(1) > 0)
    return db.loc[keep, ["ecg_id", "patient_id", "strat_fold", "filename_lr"]].reset_index(drop=True), Y[keep]


def load_raw_split(split: str, root: str = ROOT) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Unprocessed X (N, 12, 1000) in mV (cached), Y (N, 5), meta."""
    meta, Y = _select(split, root)
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, f"raw_{split}.npz")
    if os.path.exists(cp):
        z = np.load(cp)
        if np.array_equal(z["ecg_id"], meta["ecg_id"].to_numpy()):
            return z["X"], Y, meta
    X = np.stack([_read(os.path.join(root, f)) for f in tqdm(meta["filename_lr"], desc=f"load raw {split}")])
    np.savez(cp, X=X, ecg_id=meta["ecg_id"].to_numpy())
    return X, Y, meta


def load_variant(use_bandpass: bool = True, norm: str = "record", root: str = ROOT) -> dict:
    """{split: (X, Y, meta)} for train/val/test under a preprocessing variant.
    norm="record": per-record per-lead z-score (default pipeline).
    norm="dataset": one global mean/std fitted on the train split only (as in Strodthoff et al.)."""
    if use_bandpass and norm == "record":
        return {s: load_split(s, root) for s in SPLITS}
    out = {}
    for s in SPLITS:
        X, Y, meta = load_raw_split(s, root)
        X = bandpass(X) if use_bandpass else X.astype(np.float64)
        out[s] = [X, Y, meta]
    if norm == "record":
        for s in out:
            out[s][0] = zscore_record(out[s][0])
    elif norm == "dataset":
        mu, sd = out["train"][0].mean(), out["train"][0].std()
        for s in out:
            out[s][0] = (out[s][0] - mu) / sd
    else:
        raise ValueError(norm)
    return {s: (v[0].astype(np.float32), v[1], v[2]) for s, v in out.items()}


def load_filtered(root: str = ROOT) -> tuple[dict, float, float]:
    """Band-passed, NOT standardized splits {split: (X float64, Y, meta)} and the train mean/std that
    norm="dataset" uses. standardize(X, mu, sd) reproduces load_variant(norm="dataset")."""
    out = {}
    for s in SPLITS:
        X, Y, meta = load_raw_split(s, root)
        out[s] = (bandpass(X), Y, meta)
    return out, float(out["train"][0].mean()), float(out["train"][0].std())


def standardize(X: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]
