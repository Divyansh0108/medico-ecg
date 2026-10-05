import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corruptions as C  # noqa: E402
from corruptions import (ALL_FAMILIES, DROPOUT_SEC, KNOWN_FAMILIES, MODES, SEVERITIES,  # noqa: E402
                         SEVERITY_SNR, TRAIN_FAMILIES, UNSEEN_FAMILIES, Corruptor)
from data import FS, load_nstdb  # noqa: E402
from make_synthetic_data import make_synthetic, synth_ecg  # noqa: E402

# Phase 1: 100 Hz, synthetic families only. 500 Hz + NSTDB kept covered for later phases.
ADDITIVE_100 = [f for f in ALL_FAMILIES if f != "dropout"]
ADDITIVE_500 = [f for f in KNOWN_FAMILIES if f != "dropout"]
CASES = [(100, f) for f in ADDITIVE_100] + [(500, f) for f in ADDITIVE_500]


@pytest.fixture(scope="module")
def nstdb(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    make_synthetic(str(out), n_records=2, seed=0, nstdb_seconds=60.0)
    return load_nstdb(str(out / "nstdb"), target_fs=500)


def _ecg(fs):
    return synth_ecg(np.random.default_rng(0), n=10 * fs, fs=fs).astype(np.float32)


@pytest.fixture(scope="module")
def ecg():
    return _ecg(FS)


def _cor(fs, split, nstdb):
    """Phase 1 Corruptor at 100 Hz (no NSTDB); full 500 Hz Corruptor otherwise."""
    return Corruptor(None if fs == 100 else nstdb, split, fs=fs)


def _snr(x, y, a, b):
    x = x.astype(np.float64)
    n = y.astype(np.float64) - x
    return 10 * np.log10(np.var(x[a:b]) / np.mean(n[a:b] ** 2))


def test_phase1_defaults():
    assert FS == 100
    assert TRAIN_FAMILIES == ["baseline_wander", "emg"]
    assert UNSEEN_FAMILIES == ["motion_burst", "dropout"]
    assert not any(f.startswith("nstdb_") for f in ALL_FAMILIES) and "powerline" not in ALL_FAMILIES


@pytest.mark.parametrize("fs,family", CASES)
@pytest.mark.parametrize("severity,mode", list(itertools.product(SEVERITIES, MODES)))
def test_snr_on_target(nstdb, fs, family, severity, mode):
    x = _ecg(fs)
    cor = _cor(fs, "test", nstdb)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        a, b = cor._segment(len(x), mode, np.random.default_rng(seed))  # same first draws as apply
        y = cor.apply(x, family, severity, mode, rng)
        assert y.shape == x.shape and y.dtype == np.float32
        assert abs(_snr(x, y, a, b) - SEVERITY_SNR[severity]) < 0.5
        if mode == "burst":
            assert 2 * fs <= b - a <= 4 * fs
            np.testing.assert_array_equal(y[:a], x[:a])        # untouched outside burst
            np.testing.assert_array_equal(y[b:], x[b:])


@pytest.mark.parametrize("fs,mixes", [(100, (["baseline_wander", "emg"], ["baseline_wander", "emg", "motion_burst"])),
                                      (500, (["nstdb_bw", "emg"], ["baseline_wander", "nstdb_ma", "emg"]))])
@pytest.mark.parametrize("severity,mode", list(itertools.product(SEVERITIES, MODES)))
def test_mixed_snr(nstdb, fs, mixes, severity, mode):
    x = _ecg(fs)
    cor = _cor(fs, "train", nstdb)
    for fam in mixes:
        rng = np.random.default_rng(1)
        a, b = cor._segment(len(x), mode, np.random.default_rng(1))
        y = cor.apply(x, fam, severity, mode, rng)
        assert abs(_snr(x, y, a, b) - SEVERITY_SNR[severity]) < 0.5
    # the mix is not just one of its components
    y_mix = cor.apply(x, mixes[0], severity, mode, np.random.default_rng(1))
    y_one = cor.apply(x, mixes[0][0], severity, mode, np.random.default_rng(1))
    assert not np.allclose(y_mix, y_one)
    with pytest.raises(ValueError):
        cor.apply(x, ["dropout", "emg"], severity, mode, np.random.default_rng(0))


@pytest.mark.parametrize("fs", [100, 500])
@pytest.mark.parametrize("severity", SEVERITIES)
@pytest.mark.parametrize("mode", MODES)
def test_dropout_duration(nstdb, fs, severity, mode):
    x = _ecg(fs) + 10.0                  # no zeros in the input
    y = _cor(fs, "test", nstdb).apply(x, "dropout", severity, mode, np.random.default_rng(3))
    z = np.flatnonzero(y == 0)
    assert len(z) == round(DROPOUT_SEC[severity] * fs)
    assert z[-1] - z[0] + 1 == len(z)   # contiguous
    keep = np.ones(len(x), bool)
    keep[z] = False
    np.testing.assert_array_equal(y[keep], x[keep])


def test_emg_band_below_nyquist_at_100hz(ecg):
    cor = Corruptor(None, "test", fs=100)
    y = cor.apply(ecg, "emg", "severe", "whole", np.random.default_rng(0))
    n = y.astype(np.float64) - ecg
    f = np.fft.rfftfreq(len(n), 1 / 100)
    p = np.abs(np.fft.rfft(n)) ** 2
    assert p[(f >= 18) & (f <= 47)].sum() / p.sum() > 0.95


def test_determinism(ecg):
    cor = Corruptor(None, "train")
    for fam in ALL_FAMILIES:
        y1 = cor.apply(ecg, fam, "moderate", "burst", np.random.default_rng(7))
        y2 = cor.apply(ecg, fam, "moderate", "burst", np.random.default_rng(7))
        y3 = cor.apply(ecg, fam, "moderate", "burst", np.random.default_rng(8))
        np.testing.assert_array_equal(y1, y2)
        assert not np.array_equal(y1, y3)
    a, sa = cor.sample_train(ecg, np.random.default_rng(9))
    b, sb = cor.sample_train(ecg, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)
    assert sa == sb


def test_train_test_noise_disjoint():
    # Train noise = 7 Hz tone, test noise = 23 Hz tone: the corrupted spectrum reveals
    # which pool was used. A Corruptor must also never touch the other split's key.
    fs = 500
    x = _ecg(fs)
    t = np.arange(200_000) / fs
    tone = lambda f: np.stack([np.sin(2 * np.pi * f * t)] * 2, 1).astype(np.float32)  # noqa: E731
    ns_train = {k: {"train": tone(7.0)} for k in ("bw", "ma", "em")}
    ns_test = {k: {"test": tone(23.0)} for k in ("bw", "ma", "em")}
    tr, te = Corruptor(ns_train, "train", fs=fs), Corruptor(ns_test, "test", fs=fs)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    rng = np.random.default_rng(0)
    for fam in ("nstdb_bw", "nstdb_ma", "nstdb_em"):
        for cor, f0 in ((tr, 7.0), (te, 23.0)):
            y = cor.apply(x, fam, "severe", "whole", rng)
            spec = np.abs(np.fft.rfft(y.astype(np.float64) - x))
            assert f[np.argmax(spec)] == f0
    with pytest.raises(KeyError):
        Corruptor(ns_train, "test", fs=fs)


@pytest.mark.parametrize("train_families", [None, ["baseline_wander", "emg", "motion_burst"]])
def test_sample_train_families(ecg, monkeypatch, train_families):
    cor = Corruptor(None, "train", train_families=train_families)
    expected = TRAIN_FAMILIES if train_families is None else train_families
    calls = []
    real_apply = cor.apply

    def spy(x, family, severity, mode, rng):
        calls.append((family, severity, mode))
        return real_apply(x, family, severity, mode, rng)
    monkeypatch.setattr(cor, "apply", spy)
    rng = np.random.default_rng(0)
    sevs = []
    for _ in range(800):
        y, s = cor.sample_train(ecg, rng)
        sevs.append(s)
        assert y.shape == ecg.shape
    n_mix = 0
    for (fam, sev, mode), s in zip(calls, sevs):
        fams = [fam] if isinstance(fam, str) else fam
        assert set(fams) <= set(expected)
        if not isinstance(fam, str):
            n_mix += 1
            assert len(fams) == 2 and len(set(fams)) == 2
        assert SEVERITIES[s] == sev and mode in MODES
    assert abs(n_mix / len(calls) - 0.25) < 0.05
    assert set(sevs) == {0, 1, 2}
    assert {c[0] for c in calls if isinstance(c[0], str)} == set(expected)
    assert {c[2] for c in calls} == set(MODES)
    with pytest.raises(RuntimeError):
        Corruptor(None, "test").sample_train(ecg, rng)


def test_single_train_family_never_mixes(ecg):
    cor = Corruptor(None, "train", train_families=["emg"])
    rng = np.random.default_rng(0)
    for _ in range(50):
        y, _ = cor.sample_train(ecg, rng)
        assert y.shape == ecg.shape


def test_invalid_args(nstdb, ecg):
    cor = Corruptor(None, "train")
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        cor.apply(ecg, "gaussian", "mild", "whole", rng)
    with pytest.raises(ValueError):
        cor.apply(ecg, "emg", "extreme", "whole", rng)
    with pytest.raises(ValueError):
        cor.apply(ecg, "emg", "mild", "partial", rng)
    with pytest.raises(ValueError, match="NSTDB"):              # no NSTDB given
        cor.apply(ecg, "nstdb_bw", "mild", "whole", rng)
    with pytest.raises(ValueError, match="powerline"):          # 50/100 Hz not representable at 100 Hz
        cor.apply(ecg, "powerline", "mild", "whole", rng)
    with pytest.raises(ValueError, match="powerline"):
        Corruptor(nstdb, "train", fs=100, train_families=["powerline"])
    with pytest.raises(ValueError, match="dropout"):
        Corruptor(None, "train", train_families=["emg", "dropout"])
    with pytest.raises(ValueError):
        Corruptor(None, "train", train_families=[])
    with pytest.raises(ValueError):
        Corruptor(None, "val")
    assert C.SEVERITY_SNR == {"mild": 15.0, "moderate": 6.0, "severe": 0.0}
