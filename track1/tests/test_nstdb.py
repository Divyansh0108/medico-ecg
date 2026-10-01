import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nstdb  # noqa: E402
from corruptions import MODES, segment  # noqa: E402
from data import CACHE, bandpass  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    z = np.load(os.path.join(CACHE, "raw_val.npz"))
    return bandpass(z["X"][:12].astype(np.float64)), z["ecg_id"][:12]


@pytest.mark.parametrize("family", nstdb.FAMILIES)
@pytest.mark.parametrize("snr", nstdb.SNRS + [6.0])
@pytest.mark.parametrize("mode", MODES)
def test_snr_matches_target(raw, family, snr, mode):
    X, _ = raw
    for i, x in enumerate(X):
        a, b = segment(x.shape[-1], mode, np.random.default_rng(i))      # same draw as corrupt()
        y = nstdb.corrupt(x, family, snr, mode, np.random.default_rng(i))
        d = y - x
        outside = np.ones(x.shape[-1], bool)
        outside[a:b] = False
        assert np.all(d[:, outside] == 0)
        snr_m = 10 * np.log10(np.mean(x * x, -1) / np.mean(d[:, a:b] ** 2, -1))
        np.testing.assert_allclose(snr_m, snr, atol=0.5)


def test_train_eval_disjoint():
    for fam, n in nstdb.load().items():
        r = nstdb.split_ranges(n.shape[-1])
        (a0, a1), (b0, b1) = r["train"], r["eval"]
        assert a0 == 0 and b1 == n.shape[-1]
        assert b0 - a1 == int(nstdb.GAP_SEC * nstdb.FS)                 # 10 s gap
        assert not set(range(a0, a1)) & set(range(b0, b1))


@pytest.mark.parametrize("pool", ["eval", "train"])
def test_excerpts_inside_pool(raw, monkeypatch, pool):
    X, ids = raw
    seen = []
    orig = nstdb.excerpt

    def spy(fam, L, pool_, C, rng):
        n, s, ch = orig(fam, L, pool_, C, rng)
        seen.append((fam, s, L, pool_))
        return n, s, ch
    monkeypatch.setattr(nstdb, "excerpt", spy)
    for fam in nstdb.FAMILIES:
        for mode in MODES:
            for k, (x, i) in enumerate(zip(X, ids)):
                rng = nstdb.condition_rng(fam, -6.0, mode, i)
                nstdb.corrupt(x, fam, -6.0, mode, rng, pool=pool)
    if pool == "train":
        for x in X:
            nstdb.random_train(x, np.random.default_rng(0))
    assert len(seen) > 0
    for fam, s, L, p in seen:
        a, b = nstdb.split_ranges(nstdb.load()[fam].shape[-1])[pool]
        assert p == pool and a <= s and s + L <= b, (fam, s, L, pool)


def test_deterministic_and_order_independent(raw):
    X, ids = raw
    a = nstdb.corrupt_split(X[:4], ids[:4], "nstdb_mixed", 0.0, "whole")
    b = nstdb.corrupt_split(X[:4][::-1], ids[:4][::-1], "nstdb_mixed", 0.0, "whole")[::-1]
    np.testing.assert_array_equal(a, b)


def test_lead_channels_and_single_lead(raw):
    X, ids = raw
    y = nstdb.corrupt(X[0][:1], "nstdb_ma", 0.0, "whole", np.random.default_rng(0))   # 1-lead input works
    assert y.shape == (1, 1000) and np.isfinite(y).all()
