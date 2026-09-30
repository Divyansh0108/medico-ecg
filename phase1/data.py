"""PTB-XL / NSTDB loading, preprocessing and the training Dataset (spec: docs/master.md B2, B-scope).

Phase 1 runs at 100 Hz on the official PTB-XL records100 files (filename_lr);
the 500 Hz records500 path (filename_hr) is kept for later phases.

Order of operations (B2.3): corruption is applied to the RAW signal (mV),
THEN `preprocess` (band-pass 0.5-40 Hz + per-record z-score).
"""
from __future__ import annotations

import ast
import hashlib
import os
from functools import lru_cache
from math import gcd

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt
from tqdm import tqdm

FS = 100                                            # Hz, Phase 1 default (PTB-XL records100)
RECORD_SEC = 10
PTBXL_FILES = {100: "filename_lr", 500: "filename_hr"}   # fs -> ptbxl_database.csv column
N_SAMPLES = FS * RECORD_SEC                         # 1000 at the default fs
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]  # label column order

NSTDB_RECORDS = ("bw", "ma", "em")
NSTDB_FS = 360


# --------------------------------------------------------------------------- PTB-XL
def _ptbxl_labels(root: str, folds: list[int]) -> tuple[np.ndarray, pd.DataFrame]:
    """Multi-hot superclass labels + metadata for the requested folds (sorted by ecg_id).

    As in the official PTB-XL example: every scp_codes key that is a diagnostic
    statement (diagnostic == 1) is mapped to its diagnostic_class, regardless of
    likelihood. Records without any superclass are dropped.
    """
    db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    stm = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    code2cls = stm.loc[stm["diagnostic"] == 1, "diagnostic_class"].to_dict()

    db = db[db["strat_fold"].isin(folds)].sort_values("ecg_id").reset_index(drop=True)
    Y = np.zeros((len(db), len(SUPERCLASSES)), dtype=np.float32)
    for i, codes in enumerate(db["scp_codes"]):
        for code in ast.literal_eval(codes):
            cls = code2cls.get(code)
            if cls in SUPERCLASSES:
                Y[i, SUPERCLASSES.index(cls)] = 1.0
    keep = Y.sum(1) > 0
    meta = db.loc[keep, ["ecg_id", "patient_id", "strat_fold", *PTBXL_FILES.values()]].reset_index(drop=True)
    meta = meta.astype({"ecg_id": np.int64, "patient_id": np.int64, "strat_fold": np.int64})
    return Y[keep], meta


def _read_lead(path: str, lead: str, fs: int) -> np.ndarray:
    rec = wfdb.rdrecord(path, channel_names=[lead], physical=True)
    n = fs * RECORD_SEC
    if rec.fs != fs or rec.sig_len != n or rec.p_signal is None or rec.p_signal.shape[1] != 1:
        raise ValueError(f"{path}: expected lead {lead} at {fs} Hz x {n}, got "
                         f"fs={rec.fs}, len={rec.sig_len}, channels={rec.sig_name}")
    x = rec.p_signal[:, 0]
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: non-finite samples in lead {lead}")
    return x.astype(np.float32)


def load_ptbxl(root: str, folds: list[int], lead: str = "II",
               cache_dir: str | None = "../data/cache", fs: int = FS) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one lead of the PTB-XL records of `folds` at `fs` (100 -> records100, 500 -> records500).

    Returns X_raw float32 [N, fs*10] (mV, unfiltered), Y float32 [N, 5] multi-hot in
    SUPERCLASSES order, meta DataFrame (ecg_id, patient_id, strat_fold), row-aligned
    and sorted by ecg_id. Labels/meta are always recomputed from the CSVs; only the
    signals are cached, and a cache whose ecg_ids disagree with the CSVs is rebuilt.
    """
    if fs not in PTBXL_FILES:
        raise ValueError(f"fs must be one of {sorted(PTBXL_FILES)} (PTB-XL record rates), got {fs}")
    folds = sorted({int(f) for f in folds})
    if not folds or not all(1 <= f <= 10 for f in folds):
        raise ValueError(f"folds must be a non-empty subset of 1..10, got {folds}")
    Y, meta = _ptbxl_labels(root, folds)
    ids = meta["ecg_id"].to_numpy()

    cache_path = None
    if cache_dir is not None:
        root_tag = hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:8]
        name = f"ptbxl_{fs}hz_{lead}_folds{'-'.join(map(str, folds))}_{root_tag}.npz"
        cache_path = os.path.join(cache_dir, name)
        if os.path.exists(cache_path):
            with np.load(cache_path) as z:
                if np.array_equal(z["ecg_id"], ids):
                    return z["X"], Y, meta.drop(columns=list(PTBXL_FILES.values()))

    X = np.empty((len(meta), fs * RECORD_SEC), dtype=np.float32)
    for i, fn in enumerate(tqdm(meta[PTBXL_FILES[fs]], desc=f"PTB-XL {fs} Hz lead {lead} folds {folds}")):
        X[i] = _read_lead(os.path.join(root, fn), lead, fs)

    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cache_path[:-4] + ".tmp.npz"
        np.savez(tmp, X=X, ecg_id=ids)
        os.replace(tmp, cache_path)
    return X, Y, meta.drop(columns=list(PTBXL_FILES.values()))


# --------------------------------------------------------------------------- preprocessing
@lru_cache(maxsize=None)
def _bandpass_sos(fs: int) -> np.ndarray:
    return butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")


def preprocess(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """x [..., T] -> zero-phase 0.5-40 Hz Butterworth band-pass (N=4) then per-record z-score.
    At fs=100 the 40 Hz edge sits below Nyquist (50 Hz), so the same filter applies."""
    y = sosfiltfilt(_bandpass_sos(fs), np.asarray(x, dtype=np.float64), axis=-1)
    y = (y - y.mean(-1, keepdims=True)) / (y.std(-1, keepdims=True) + 1e-6)
    return y.astype(np.float32)


# --------------------------------------------------------------------------- NSTDB
def load_nstdb(root: str, target_fs: int = FS, train_frac: float = 0.6) -> dict[str, dict[str, np.ndarray]]:
    """{"bw"|"ma"|"em": {"train": [T, 2], "test": [T, 2]}} float32, resampled to target_fs.

    Each record is split IN TIME at the original 360 Hz rate (first `train_frac`
    -> train, remainder -> test) and each part is resampled independently, so the
    resampling filter never mixes samples across the split (strictly disjoint).
    """
    out = {}
    for name in NSTDB_RECORDS:
        rec = wfdb.rdrecord(os.path.join(root, name), physical=True)
        if rec.fs != NSTDB_FS or rec.p_signal.shape[1] != 2:
            raise ValueError(f"NSTDB {name}: expected 2 channels at {NSTDB_FS} Hz")
        sig = rec.p_signal.astype(np.float64)
        cut = int(round(train_frac * len(sig)))
        g = gcd(int(target_fs), NSTDB_FS)
        up, down = int(target_fs) // g, NSTDB_FS // g
        out[name] = {part: resample_poly(seg, up, down, axis=0).astype(np.float32)
                     for part, seg in (("train", sig[:cut]), ("test", sig[cut:]))}
    return out


# --------------------------------------------------------------------------- Dataset
class ECGDataset(torch.utils.data.Dataset):
    """Single-lead dataset; corruption (raw) -> preprocess, on the fly per sample.

    unpaired: {"x": [1, T], "y": [5], "idx"}; with prob p_corrupt the raw signal is
              first corrupted by corruptor.sample_train().
    paired:   {"x_clean", "x_corr": [1, T], "y", "idx", "severity"}; x_corr is always
              corrupted with sample_train(), x_clean never.
    RNG: `self.rng`, seeded from `seed` in the main process; in DataLoader workers
    `worker_init_fn` reseeds it from (seed, torch worker seed). The torch worker seed
    is base_seed + worker_id, where base_seed is drawn from the loader's torch RNG
    for every new iterator, so streams differ across workers and epochs yet are
    reproducible under torch.manual_seed.
    """

    def __init__(self, X_raw, Y, corruptor=None, p_corrupt: float = 0.5,
                 paired: bool = False, seed: int = 0, fs: int = FS):
        if len(X_raw) != len(Y):
            raise ValueError("X_raw and Y must have the same length")
        if paired and corruptor is None:
            raise ValueError("paired mode requires a corruptor")
        if not 0.0 <= p_corrupt <= 1.0:
            raise ValueError("p_corrupt must be in [0, 1]")
        self.X_raw = X_raw
        self.Y = np.asarray(Y, dtype=np.float32)
        self.corruptor = corruptor
        self.p_corrupt = p_corrupt
        self.paired = paired
        self.seed = seed
        self.fs = int(fs)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X_raw)

    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(preprocess(x, self.fs)[None])

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        x = np.asarray(self.X_raw[idx], dtype=np.float32)
        y = torch.from_numpy(self.Y[idx].copy())
        if self.paired:
            x_corr, sev = self.corruptor.sample_train(x, self.rng)
            return {"x_clean": self._t(x), "x_corr": self._t(x_corr), "y": y,
                    "idx": idx, "severity": int(sev)}
        if self.corruptor is not None and self.rng.random() < self.p_corrupt:
            x, _ = self.corruptor.sample_train(x, self.rng)
        return {"x": self._t(x), "y": y, "idx": idx}


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker its own reproducible RNG stream."""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    while not isinstance(ds, ECGDataset) and hasattr(ds, "dataset"):  # e.g. Subset
        ds = ds.dataset
    if isinstance(ds, ECGDataset):
        ds.rng = np.random.default_rng([ds.seed, info.seed])
