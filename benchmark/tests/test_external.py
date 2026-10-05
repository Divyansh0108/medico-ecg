"""Checks for the external-dataset pipelines (RULES2.md sections 3 and 5)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from external.single_lead_infer import h1, h2, windows  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BQ = os.path.join(HERE, "results", "robustness", "butqdb")


def test_windows_cover_and_pad():
    for T in [450, 900, 1000, 1001, 1499, 3000, 6123]:
        x = np.arange(T, dtype=float)
        w = windows(x)
        assert w.shape[1] == 1000
        if T >= 1000:
            assert w[0, 0] == 0 and w[-1, -1] == T - 1                   # tail covered
            assert np.all(np.diff(w[:, 0])[:-1] == 500)
        else:
            assert w.shape[0] == 1 and np.array_equal(w[0, :T], x)       # reflect pad keeps the signal


def test_heuristics_direction():
    rng = np.random.default_rng(0)
    t = np.arange(1000) / 100
    clean = np.sin(2 * np.pi * 1.2 * t) ** 21                             # spiky, low-frequency
    noisy = clean + 0.5 * rng.standard_normal(1000)
    assert h1(noisy) > h1(clean)
    assert h2(np.r_[np.zeros(999), 50.0]) > h2(clean)


@pytest.mark.skipif(not os.path.exists(os.path.join(BQ, "windows.npz")), reason="run external/butqdb.py prep first")
def test_butqdb_windows_single_class():
    """Every kept window lies inside one consensus class in {1, 2, 3} (rebuilt from the raw annotation CSV)."""
    z = np.load(os.path.join(BQ, "windows.npz"))
    log = pd.read_csv(os.path.join(BQ, "windows_log.csv"), dtype={"record": str})
    assert len(z["cls"]) == log.kept.sum() and set(np.unique(z["cls"])) <= {1, 2, 3}
    assert len(np.unique(z["subject"])) == 15
    r = "114001"                                                          # small fully-checked example
    a = pd.read_csv(os.path.join(HERE, "..", "data", "butqdb", r, f"{r}_ANN.csv"), header=None).iloc[:, 9:12].dropna().astype(int)
    lab = np.zeros(int(a[10].max()), np.int8)
    for s, e, q in a.itertuples(index=False):
        lab[s - 1:e] = q
    nw = len(lab) // 10000
    L = lab[:nw * 10000].reshape(nw, 10000)
    keep = (L.min(1) == L.max(1)) & (L.min(1) > 0)
    assert keep.sum() == int(log.set_index("record").loc[r, "kept"])
    assert np.array_equal(L.min(1)[keep], z["cls"][z["rec"] == r])
