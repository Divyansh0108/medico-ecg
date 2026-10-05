"""Track 2 corruptions for 12-lead PTB-XL at 100 Hz.

Order: raw -> band-pass (data.bandpass) -> ADD NOISE HERE -> dataset-level standardization with the
TRAIN mean/std. The noise is never filtered, and there is no per-record z-score to rescale it away.

All families are additive and applied per record to all 12 leads. Each lead has its own noise
realization, scaled to that lead's power:
    SNR_dB = 10 log10(P_signal / P_noise),
where P_signal = mean square of the filtered lead over the whole record and P_noise = mean square of
the added noise over the samples it covers (whole record, or the 2-4 s burst).
  seen (training aug) : baseline_wander, emg
  unseen (test only)  : motion_burst, dropout, powerline
  mixed               : 2 distinct families of the 5, each at half the target power, summed and
                        rescaled so the sum hits the target exactly
Modes: whole record, or one 2-4 s burst at a random position.

Notes on two families at fs = 100 Hz:
- powerline: 50 Hz is the Nyquist frequency here. A pure 50 Hz tone is an alternating sequence whose
  amplitude depends on the phase, so the mains frequency drifts in U[49.9, 50.1] Hz. The result is an
  alternating tone with a slow beat envelope.
- dropout: flat-line gaps (0.05-0.5 s) at a random DC offset (lost contact with a baseline jump). The
  noise is (offset - x) on each gap and 0 elsewhere. Gaps are added while they keep the target
  reachable, then the offset scale solves the SNR exactly.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from loaders.ptbxl import FS

SEEN = ["baseline_wander", "emg"]
UNSEEN = ["motion_burst", "dropout", "powerline"]
FAMILIES = SEEN + UNSEEN
ALL_CONDITION_FAMILIES = FAMILIES + ["mixed"]
SNRS = [15.0, 6.0, 0.0, -6.0]
MODES = ["whole", "burst"]
BURST_SEC = (2.0, 4.0)
TEST_SEED = 20261001
TRAIN_P = 0.5
TRAIN_SNR = (0.0, 20.0)


@lru_cache(maxsize=None)
def _sos(lo: float, hi: float) -> np.ndarray:
    return butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")


def _unit(n: np.ndarray) -> np.ndarray:
    """Per lead: zero mean, unit mean power (rows with zero power stay zero)."""
    n = n - n.mean(-1, keepdims=True)
    p = np.mean(n * n, -1, keepdims=True)
    return np.divide(n, np.sqrt(p), out=np.zeros_like(n), where=p > 0)


def segment(T: int, mode: str, rng: np.random.Generator) -> tuple[int, int]:
    if mode == "whole":
        return 0, T
    if mode != "burst":
        raise ValueError(mode)
    L = min(T, int(rng.integers(int(BURST_SEC[0] * FS), int(BURST_SEC[1] * FS) + 1)))
    s = int(rng.integers(0, T - L + 1))
    return s, s + L


def _shape(family: str, C: int, L: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """(C, L) unit-power noise, independent per lead."""
    t = np.arange(L) / FS
    if family == "baseline_wander":
        k = 3
        f, a, ph = rng.uniform(0.05, 0.5, (C, k, 1)), rng.uniform(0.5, 1.0, (C, k, 1)), rng.uniform(0, 2 * np.pi, (C, k, 1))
        n = (a * np.sin(2 * np.pi * f * t + ph)).sum(1)
    elif family == "emg":
        n = sosfiltfilt(_sos(20.0, 45.0), rng.standard_normal((C, L)), axis=-1)
    elif family == "powerline":
        f, ph = rng.uniform(49.9, 50.1), rng.uniform(0, 2 * np.pi, (C, 1))
        n = np.sin(2 * np.pi * f * t + ph)
    elif family == "motion_burst":
        # band-limited 1-10 Hz noise plus a step offset under a Tukey envelope (as in Phase 1);
        # burst mode: one event over the whole burst; whole mode: 2-4 events of 0.5-1.5 s per lead.
        bp = _unit(sosfiltfilt(_sos(1.0, 10.0), rng.standard_normal((C, L)), axis=-1))
        n = np.zeros((C, L))
        for c in range(C):
            if mode == "burst":
                events = [(0, L)]
            else:
                events = []
                for _ in range(int(rng.integers(2, 5))):
                    d = min(L, int(rng.integers(int(0.5 * FS), int(1.5 * FS) + 1)))
                    events.append((int(rng.integers(0, L - d + 1)), d))
            for s, d in events:
                step = rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 3.0)
                n[c, s:s + d] += (bp[c, s:s + d] + step) * tukey(d, 0.5)
    else:
        raise ValueError(f"unknown family {family!r}")
    return _unit(n)


def _dropout(xs: np.ndarray, p_t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(C, L) dropout noise with mean power exactly p_t per lead over the segment xs."""
    C, L = xs.shape
    out = np.zeros_like(xs)
    for c in range(C):
        x, P = xs[c], p_t[c] * L                     # P = target total energy on this lead
        if P <= 0:
            continue
        used = np.zeros(L, bool)
        gaps = []                                    # (start, stop, offset sign*size)
        A = B = E = 0.0                              # sum c^2, sum c*x, sum x^2 over gap samples
        for _ in range(60):
            if used.sum() >= 0.3 * L:
                break
            d = min(L, int(rng.integers(max(1, int(0.05 * FS)), int(0.5 * FS) + 1)))
            s = int(rng.integers(0, L - d + 1))
            if used[s:s + d].any():
                continue
            e = float(np.sum(x[s:s + d] ** 2))
            if E + e > 0.5 * P:                      # keep the target reachable with a positive offset
                continue
            o = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5)
            gaps.append((s, s + d, o))
            used[s:s + d] = True
            A, B, E = A + o * o * d, B + o * float(x[s:s + d].sum()), E + e
        if not gaps:                                 # fallback: one 1-sample gap where |x| is smallest
            s = int(np.argmin(np.abs(x)))
            o = rng.choice([-1.0, 1.0])
            gaps, A, B, E = [(s, s + 1, o)], 1.0, o * float(x[s]), float(x[s] ** 2)
        # solve A k^2 - 2 B k + E = P for the offset scale k > 0
        k = (B + np.sqrt(max(B * B - A * (E - P), 0.0))) / A
        for s, e_, o in gaps:
            out[c, s:e_] = o * k - x[s:e_]
    return out


def _noise(family: str, xs: np.ndarray, p_t: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if family == "dropout":
        return _dropout(xs, p_t, rng)
    return _shape(family, xs.shape[0], xs.shape[1], mode, rng) * np.sqrt(p_t)[:, None]


def corrupt(x: np.ndarray, family: str, snr_db: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    """x: (12, T) band-passed record (not standardized) -> corrupted copy (float64)."""
    x = np.asarray(x, dtype=np.float64)
    p_sig = np.mean(x * x, -1)
    p_t = p_sig / 10.0 ** (snr_db / 10.0)
    a, b = segment(x.shape[-1], mode, rng)
    xs = x[:, a:b]
    if family == "mixed":
        f1, f2 = rng.choice(FAMILIES, 2, replace=False)
        n = _noise(f1, xs, p_t / 2, mode, rng) + _noise(f2, xs, p_t / 2, mode, rng)
        p = np.mean(n * n, -1)
        n *= np.sqrt(np.divide(p_t, p, out=np.zeros_like(p), where=p > 0))[:, None]
    else:
        n = _noise(family, xs, p_t, mode, rng)
    y = x.copy()
    y[:, a:b] += n
    return y


def condition_rng(family: str, snr_db: float, mode: str, ecg_id: int, seed: int = TEST_SEED) -> np.random.Generator:
    """Fixed RNG per (condition, record): every model sees identical noisy records."""
    return np.random.default_rng([seed, ALL_CONDITION_FAMILIES.index(family), int(round(snr_db)) + 100,
                                  MODES.index(mode), int(ecg_id)])


def corrupt_split(X: np.ndarray, ecg_ids: np.ndarray, family: str, snr_db: float, mode: str,
                  seed: int = TEST_SEED) -> np.ndarray:
    """X: (N, 12, T) band-passed -> corrupted (N, 12, T) float64, deterministic per record."""
    return np.stack([corrupt(x, family, snr_db, mode, condition_rng(family, snr_db, mode, i, seed))
                     for x, i in zip(X, ecg_ids)])


def conditions(snrs) -> list[tuple[str, float, str]]:
    return [(f, float(s), m) for f in ALL_CONDITION_FAMILIES for s in snrs for m in MODES]


def cond_name(family: str, snr_db: float, mode: str) -> str:
    return f"{family}|{snr_db:+.0f}|{mode}"


def random_seen(x: np.ndarray, rng: np.random.Generator, families=SEEN) -> tuple[np.ndarray, float]:
    """Training corruption: random SEEN family, SNR ~ U[0, 20] dB, random mode. Returns (y, snr).
    families=FAMILIES (--aug-families all, RULES3.md) draws from all five families instead."""
    fam = families[int(rng.integers(len(families)))]
    snr = float(rng.uniform(*TRAIN_SNR))
    mode = MODES[int(rng.integers(len(MODES)))]
    return corrupt(x, fam, snr, mode, rng), snr
