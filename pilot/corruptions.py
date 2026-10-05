"""On-the-fly ECG corruptions (spec: docs/master.md B3).

All corruptions act on the RAW single-lead signal (mV) before `data.preprocess`.
Additive families: a unit-power, zero-mean noise is drawn for the affected
segment (whole record, or one contiguous 2-4 s burst), mixed families are summed
and the SUM is scaled so that
    SNR = 10 log10(P_signal / P_noise),
with P_signal the variance of the clean segment and P_noise the mean power of the
added noise over the same segment. Dropout zeroes a segment of DROPOUT_SEC.
Every draw comes from the `rng` passed in, so results are deterministic given it.

Phase 1 scope (docs/master.md B-scope): 100 Hz, synthetic families only. The NSTDB
families and powerline remain implemented for later phases; the Corruptor refuses
them when their inputs are missing (no NSTDB) or unrepresentable (powerline needs
its 100 Hz harmonic below Nyquist, i.e. fs > 200).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from data import FS

SEVERITY_SNR = {"mild": 15.0, "moderate": 6.0, "severe": 0.0}      # dB
DROPOUT_SEC = {"mild": 0.5, "moderate": 1.0, "severe": 2.0}
SEVERITIES = ["mild", "moderate", "severe"]                        # index = severity int
TRAIN_FAMILIES = ["baseline_wander", "emg"]         # Phase 1 defaults (configs/phase1.yaml)
UNSEEN_FAMILIES = ["motion_burst", "dropout"]
ALL_FAMILIES = TRAIN_FAMILIES + UNSEEN_FAMILIES
NSTDB_FAMILIES = ["nstdb_bw", "nstdb_ma", "nstdb_em"]
KNOWN_FAMILIES = NSTDB_FAMILIES + ["baseline_wander", "emg", "powerline", "motion_burst", "dropout"]
EMG_BAND = (20.0, 150.0)              # Hz; upper edge capped at 0.45*fs (45 Hz at 100 Hz)
MODES = ["whole", "burst"]            # burst = one contiguous 2-4 s segment

BURST_SEC = (2.0, 4.0)                # burst-mode segment length range
P_MIX = 0.25                          # sample_train: probability of a 2-family mix
_NSTDB = {"nstdb_bw": "bw", "nstdb_ma": "ma", "nstdb_em": "em"}


@lru_cache(maxsize=None)
def _sos(lo: float, hi: float, fs: int) -> np.ndarray:
    return butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")


def _unit(n: np.ndarray) -> np.ndarray:
    """Zero-mean, unit mean-power version of n."""
    n = n - n.mean()
    p = np.mean(n * n)
    return n / np.sqrt(p) if p > 0 else n


class Corruptor:
    def __init__(self, nstdb: dict | None, split: str, fs: int = FS,
                 train_families: list[str] | None = None):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.split = split
        self.fs = int(fs)
        # Keep ONLY this split's excerpts: train and test noise can never mix.
        self.noise = ({} if nstdb is None else
                      {k: np.ascontiguousarray(nstdb[k][split], dtype=np.float64) for k in _NSTDB.values()})
        self.train_families = list(TRAIN_FAMILIES if train_families is None else train_families)
        if not self.train_families:
            raise ValueError("train_families must not be empty")
        for f in self.train_families:
            if f == "dropout":
                raise ValueError("dropout cannot be a training family (it cannot be mixed)")
            self._check_family(f)

    def _check_family(self, family: str) -> None:
        if family not in KNOWN_FAMILIES:
            raise ValueError(f"unknown family {family!r}; expected one of {KNOWN_FAMILIES}")
        if family in _NSTDB and not self.noise:
            raise ValueError(f"family {family!r} needs NSTDB noise, but no NSTDB records were given")
        if family == "powerline" and self.fs <= 200:
            raise ValueError(f"powerline (50 Hz + 100 Hz harmonic) is not representable at fs={self.fs}")

    # ------------------------------------------------------------------ helpers
    def _segment(self, n: int, mode: str, rng: np.random.Generator) -> tuple[int, int]:
        """(start, stop) of the affected segment."""
        if mode == "whole":
            return 0, n
        L = min(n, int(rng.integers(int(BURST_SEC[0] * self.fs), int(BURST_SEC[1] * self.fs) + 1)))
        s = int(rng.integers(0, n - L + 1))
        return s, s + L

    def _noise(self, family: str, L: int, mode: str, rng: np.random.Generator) -> np.ndarray:
        """Zero-mean unit-power noise of length L for one additive family."""
        fs = self.fs
        t = np.arange(L) / fs
        if family in _NSTDB:
            arr = self.noise[_NSTDB[family]]
            if len(arr) < L:
                raise ValueError(f"NSTDB {family} {self.split} excerpt shorter than {L} samples")
            ch = int(rng.integers(arr.shape[1]))
            s = int(rng.integers(0, len(arr) - L + 1))
            n = arr[s:s + L, ch]
        elif family == "baseline_wander":
            k = int(rng.integers(1, 4))
            f = rng.uniform(0.05, 0.5, k)
            a = rng.uniform(0.5, 1.0, k)
            ph = rng.uniform(0, 2 * np.pi, k)
            n = (a[:, None] * np.sin(2 * np.pi * f[:, None] * t + ph[:, None])).sum(0)
        elif family == "emg":
            n = sosfiltfilt(_sos(EMG_BAND[0], min(EMG_BAND[1], 0.45 * fs), fs), rng.standard_normal(L))
        elif family == "powerline":
            ph = rng.uniform(0, 2 * np.pi, 2)
            h = rng.uniform(0.05, 0.2)
            n = np.sin(2 * np.pi * 50 * t + ph[0]) + h * np.sin(2 * np.pi * 100 * t + ph[1])
        elif family == "motion_burst":
            # Low-frequency (1-10 Hz) band-limited noise plus a step offset, under a
            # Tukey envelope. burst mode: one event spanning the segment;
            # whole mode: 2-4 events of 0.5-1.5 s at random positions.
            bp = _unit(sosfiltfilt(_sos(1.0, 10.0, fs), rng.standard_normal(L)))
            n = np.zeros(L)
            if mode == "burst":
                events = [(0, L)]
            else:
                events = []
                for _ in range(int(rng.integers(2, 5))):
                    d = min(L, int(rng.integers(int(0.5 * fs), int(1.5 * fs) + 1)))
                    s = int(rng.integers(0, L - d + 1))
                    events.append((s, d))
            for s, d in events:
                step = rng.choice([-1.0, 1.0]) * rng.uniform(1.0, 3.0)
                n[s:s + d] += (bp[s:s + d] + step) * tukey(d, 0.5)
        else:
            raise ValueError(f"unknown additive family {family!r}")
        return _unit(n)

    # ------------------------------------------------------------------ API
    def apply(self, x: np.ndarray, family: str | list[str], severity: str,
              mode: str, rng: np.random.Generator) -> np.ndarray:
        """RAW x [T] -> corrupted RAW [T] (float32). A list of families = mixed."""
        if severity not in SEVERITY_SNR:
            raise ValueError(f"unknown severity {severity!r}")
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        families = [family] if isinstance(family, str) else list(family)
        if not families:
            raise ValueError("empty family list")
        for f in families:
            self._check_family(f)
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("apply expects a single 1-D record")
        y = x.astype(np.float64)

        if "dropout" in families:
            if len(families) > 1:
                raise ValueError("dropout cannot be mixed with additive families")
            d = min(len(y), int(round(DROPOUT_SEC[severity] * self.fs)))
            s = int(rng.integers(0, len(y) - d + 1))
            y[s:s + d] = 0.0
            return y.astype(np.float32)

        a, b = self._segment(len(y), mode, rng)
        L = b - a
        noise = sum(self._noise(f, L, mode, rng) for f in families)
        noise = noise - noise.mean()
        p_sig = np.var(y[a:b])
        p_noise = np.mean(noise * noise)
        if p_sig > 0 and p_noise > 0:
            noise *= np.sqrt(p_sig / (10.0 ** (SEVERITY_SNR[severity] / 10.0)) / p_noise)
        else:
            noise[:] = 0.0
        y[a:b] += noise
        return y.astype(np.float32)

    def sample_train(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        """Random training corruption: one self.train_families member, or (p=0.25, if there
        are >= 2) a mix of two distinct ones; uniform severity and mode.
        Returns (x_corr, severity_int)."""
        if self.split != "train":
            raise RuntimeError("sample_train must only be used with the 'train' noise split")
        tf = self.train_families
        if len(tf) >= 2 and rng.random() < P_MIX:
            fam = [tf[i] for i in rng.choice(len(tf), 2, replace=False)]
        else:
            fam = tf[int(rng.integers(len(tf)))]
        sev = int(rng.integers(len(SEVERITIES)))
        mode = MODES[int(rng.integers(len(MODES)))]
        return self.apply(x, fam, SEVERITIES[sev], mode, rng), sev
