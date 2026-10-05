import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from noise.synthetic import (ALL_CONDITION_FAMILIES, FAMILIES, MODES, SNRS, corrupt, corrupt_split,  # noqa: E402
                         random_seen, segment)
from loaders.ptbxl import CACHE, bandpass, standardize  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    z = np.load(os.path.join(CACHE, "raw_val.npz"))
    return z["X"][:12].astype(np.float64), z["ecg_id"][:12]


def measured_snr(x, y, mode_seg):
    a, b = mode_seg
    n = (y - x)[:, a:b]
    return 10 * np.log10(np.mean(x * x, -1) / np.mean(n * n, -1))


@pytest.mark.parametrize("family", ALL_CONDITION_FAMILIES)
@pytest.mark.parametrize("snr", SNRS)
@pytest.mark.parametrize("mode", MODES)
def test_snr_matches_target(raw, family, snr, mode):
    X, _ = raw
    for i, x in enumerate(bandpass(X)):
        seg = segment(x.shape[-1], mode, np.random.default_rng(i))       # same draws as corrupt()
        y = corrupt(x, family, snr, mode, np.random.default_rng(i))
        d = y - x
        outside = np.ones(x.shape[-1], bool)
        outside[seg[0]:seg[1]] = False
        assert np.all(d[:, outside] == 0), "noise outside the burst"
        np.testing.assert_allclose(measured_snr(x, y, seg), snr, atol=0.5)


def test_noise_added_after_filtering(raw):
    """pipeline = standardize(bandpass(raw) + noise): the noise keeps content the band-pass removes."""
    X, _ = raw
    xf = bandpass(X[0])
    mu, sd = 0.01, 0.2
    for fam, band in (("powerline", lambda f: f > 45), ("baseline_wander", lambda f: f <= 0.7)):
        y = corrupt(xf, fam, 0.0, "whole", np.random.default_rng(0))
        dz = standardize(y, mu, sd).astype(np.float64) - standardize(xf, mu, sd)
        np.testing.assert_allclose(dz, (y - xf) / sd, atol=1e-4)            # noise enters unfiltered
        spec = np.abs(np.fft.rfft(y - xf, axis=-1)) ** 2
        f = np.fft.rfftfreq(xf.shape[-1], 1 / 100)
        assert spec[:, band(f)].sum() / spec.sum() > 0.85                    # outside the 0.5-40 Hz pass band (10 s: 0.1 Hz bins leak)
        refilt = bandpass(y) - bandpass(xf)                                   # filtering after would remove it
        assert np.mean(refilt ** 2) < 0.2 * np.mean((y - xf) ** 2)


def test_no_nans_including_flat_lead(raw):
    X, ids = raw
    Xf = bandpass(X[:4])
    Xf[0, 3] = 0.0                                                           # a flat lead
    for fam in ALL_CONDITION_FAMILIES:
        for snr in SNRS:
            for mode in MODES:
                Y = corrupt_split(Xf, ids[:4], fam, snr, mode)
                assert np.isfinite(Y).all()
                assert np.all(Y[0, 3] == 0)                                  # zero-power lead gets no noise


def test_deterministic_for_same_seed(raw):
    X, ids = raw
    Xf = bandpass(X[:4])
    for fam in ALL_CONDITION_FAMILIES:
        a = corrupt_split(Xf, ids[:4], fam, 0.0, "burst")
        b = corrupt_split(Xf, ids[:4], fam, 0.0, "burst")
        c = corrupt_split(Xf, ids[:4], fam, 0.0, "burst", seed=1)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
    y1, s1 = random_seen(Xf[0], np.random.default_rng(5))
    y2, s2 = random_seen(Xf[0], np.random.default_rng(5))
    np.testing.assert_array_equal(y1, y2)
    assert s1 == s2 and 0 <= s1 <= 20


def test_order_independent(raw):
    """A record's noise depends on its ecg_id, not on its position in the split."""
    X, ids = raw
    Xf = bandpass(X[:4])
    a = corrupt_split(Xf, ids[:4], "mixed", 6.0, "whole")
    b = corrupt_split(Xf[::-1], ids[:4][::-1], "mixed", 6.0, "whole")[::-1]
    np.testing.assert_array_equal(a, b)


def test_families_differ(raw):
    X, _ = raw
    xf = bandpass(X[0])
    ns = [corrupt(xf, f, 0.0, "whole", np.random.default_rng(0)) - xf for f in FAMILIES]
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            assert not np.allclose(ns[i], ns[j])
