"""Real recorded noise from the MIT-BIH Noise Stress Test Database (rules: results/track2/RULES2.md, section 1).

Records bw (baseline wander), ma (muscle artifact), em (electrode motion): 2 channels, 360 Hz.
Noise: resample 360 -> 100 Hz (polyphase) -> data.bandpass (the ECG filter). Each record is split in
time: TRAIN = first 60 %, a 10 s gap, EVAL = the rest. Eval noise is drawn only from EVAL.
Order on the ECG side is the same as corruptions.py: band-pass(ECG) -> add noise -> standardize.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import wfdb
from scipy.signal import resample_poly

from corruptions import MODES, TRAIN_SNR, _unit, segment
from data import FS, bandpass

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "nstdb")
RECORDS = {"nstdb_bw": "bw", "nstdb_ma": "ma", "nstdb_em": "em"}
SINGLE = list(RECORDS)
FAMILIES = SINGLE + ["nstdb_mixed"]
SNRS = [0.0, -6.0]
TRAIN_FRAC, GAP_SEC = 0.6, 10.0
TEST_SEED = 20261001


def split_ranges(T: int) -> dict[str, tuple[int, int]]:
    """[start, stop) of the TRAIN and EVAL pools for a record of T samples (100 Hz)."""
    a = int(np.floor(TRAIN_FRAC * T))
    tr, ev = (0, a), (a + int(GAP_SEC * FS), T)
    assert tr[1] <= ev[0] and ev[0] < ev[1], (tr, ev)        # disjoint, non-empty
    return {"train": tr, "eval": ev}


@lru_cache(maxsize=None)
def load(verbose: bool = False) -> dict[str, np.ndarray]:
    """{family: (2, T) float64 noise at 100 Hz, band-passed}."""
    out = {}
    for fam, rec in RECORDS.items():
        r = wfdb.rdrecord(os.path.join(ROOT, rec), physical=True)
        x = r.p_signal.T.astype(np.float64)
        if verbose:
            print(f"{rec}: fs={r.fs} Hz, channels={r.n_sig} {r.sig_name}, length={r.sig_len} samples "
                  f"({r.sig_len / r.fs / 60:.2f} min)")
        assert r.fs == 360 and r.n_sig == 2
        y = bandpass(resample_poly(x, 5, 18, axis=-1))
        if verbose:
            print(f"  -> 100 Hz: {y.shape[-1]} samples; pools {split_ranges(y.shape[-1])}")
        out[fam] = y
    return out


def excerpt(fam: str, L: int, pool: str, C: int, rng: np.random.Generator) -> tuple[np.ndarray, int, np.ndarray]:
    """(C, L) unit-power noise: one excerpt of the pool, one random channel per lead. Returns (n, start, ch)."""
    n = load()[fam]
    a, b = split_ranges(n.shape[-1])[pool]
    s = int(rng.integers(a, b - L + 1))
    ch = rng.integers(0, n.shape[0], C)
    return _unit(n[ch, s:s + L]), s, ch


def _shape(fam: str, C: int, L: int, pool: str, rng: np.random.Generator) -> np.ndarray:
    if fam != "nstdb_mixed":
        return excerpt(fam, L, pool, C, rng)[0]
    f1, f2 = rng.choice(SINGLE, 2, replace=False)
    return _unit(excerpt(f1, L, pool, C, rng)[0] + excerpt(f2, L, pool, C, rng)[0])


def corrupt(x: np.ndarray, family: str, snr_db: float, mode: str, rng: np.random.Generator,
            pool: str = "eval") -> np.ndarray:
    """x: (C, T) band-passed record (not standardized) -> corrupted copy (float64)."""
    x = np.asarray(x, dtype=np.float64)
    p_t = np.mean(x * x, -1) / 10.0 ** (snr_db / 10.0)
    a, b = segment(x.shape[-1], mode, rng)
    n = _shape(family, x.shape[0], b - a, pool, rng) * np.sqrt(p_t)[:, None]
    y = x.copy()
    y[:, a:b] += n
    return y


def condition_rng(family: str, snr_db: float, mode: str, ecg_id: int, seed: int = TEST_SEED) -> np.random.Generator:
    return np.random.default_rng([seed, FAMILIES.index(family), int(round(snr_db)) + 100, MODES.index(mode),
                                  int(ecg_id), 7])


def corrupt_split(X: np.ndarray, ecg_ids: np.ndarray, family: str, snr_db: float, mode: str,
                  seed: int = TEST_SEED) -> np.ndarray:
    return np.stack([corrupt(x, family, snr_db, mode, condition_rng(family, snr_db, mode, i, seed))
                     for x, i in zip(X, ecg_ids)])


def conditions(snrs=SNRS) -> list[tuple[str, float, str]]:
    return [(f, float(s), m) for f in FAMILIES for s in snrs for m in MODES]


def random_train(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Training corruption from the TRAIN pool: random NSTDB family, SNR ~ U[0, 20] dB, random mode."""
    fam = FAMILIES[int(rng.integers(len(FAMILIES)))]
    snr = float(rng.uniform(*TRAIN_SNR))
    mode = MODES[int(rng.integers(len(MODES)))]
    return corrupt(x, fam, snr, mode, rng, pool="train"), snr


if __name__ == "__main__":
    load(verbose=True)
