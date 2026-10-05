import ast
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import wfdb
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data  # noqa: E402
from corruptions import Corruptor  # noqa: E402
from data import FS, N_SAMPLES, SUPERCLASSES, ECGDataset, load_nstdb, load_ptbxl, preprocess  # noqa: E402
from make_synthetic_data import make_synthetic  # noqa: E402

# Independent reference mapping (real scp_statements.csv) for the codes the generator uses.
REF_CLASS = {"NORM": "NORM", "IMI": "MI", "AMI": "MI", "NDT": "STTC", "ISC_": "STTC",
             "CLBBB": "CD", "IRBBB": "CD", "LVH": "HYP", "RVH": "HYP"}


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    make_synthetic(str(out), n_records=200, seed=0, nstdb_seconds=60.0)
    return out


@pytest.fixture(scope="module")
def ptb(syn):
    return str(syn / "ptb-xl")


def _expected(ptb_root, folds):
    db = pd.read_csv(os.path.join(ptb_root, "ptbxl_database.csv"))
    db = db[db.strat_fold.isin(folds)].sort_values("ecg_id")
    ids, ys = [], []
    for eid, codes in zip(db.ecg_id, db.scp_codes):
        y = np.zeros(5, np.float32)
        for c in ast.literal_eval(codes):
            if c in REF_CLASS:
                y[SUPERCLASSES.index(REF_CLASS[c])] = 1
        if y.any():
            ids.append(eid)
            ys.append(y)
    return np.array(ids), np.array(ys)


# --------------------------------------------------------------------------- PTB-XL
def test_label_mapping_and_drop(ptb):
    folds = list(range(1, 11))
    X, Y, meta = load_ptbxl(ptb, folds, cache_dir=None)
    ids, Yexp = _expected(ptb, folds)
    assert X.shape == (len(ids), N_SAMPLES) and X.dtype == np.float32
    assert Y.dtype == np.float32 and Y.shape == (len(ids), 5)
    np.testing.assert_array_equal(meta.ecg_id.to_numpy(), ids)
    np.testing.assert_array_equal(Y, Yexp)
    assert list(meta.columns) == ["ecg_id", "patient_id", "strat_fold"]
    # all superclasses present, multi-label present, no-superclass records dropped
    assert (Y.sum(0) > 0).all() and (Y.sum(1) > 1).any() and (Y.sum(1) > 0).all()
    n_total = len(pd.read_csv(os.path.join(ptb, "ptbxl_database.csv")))
    assert len(Y) < n_total
    # a zero-likelihood diagnostic code still counts (pattern {"IRBBB":100,"RVH":0,"NDT":0})
    db = pd.read_csv(os.path.join(ptb, "ptbxl_database.csv")).set_index("ecg_id")
    row = [i for i, e in enumerate(ids) if "RVH" in db.loc[e, "scp_codes"]][0]
    np.testing.assert_array_equal(Y[row], [0, 0, 1, 1, 1])


@pytest.mark.parametrize("fs,folder,suffix", [(100, "records100", "lr"), (500, "records500", "hr")])
def test_signal_matches_wfdb_lead_ii(ptb, fs, folder, suffix):
    X, _, meta = load_ptbxl(ptb, [3], cache_dir=None, fs=fs)
    assert X.shape[1] == 10 * fs
    eid = int(meta.ecg_id.iloc[0])
    rec = wfdb.rdrecord(os.path.join(ptb, f"{folder}/00000/{eid:05d}_{suffix}"))
    assert rec.fs == fs
    np.testing.assert_allclose(X[0], rec.p_signal[:, rec.sig_name.index("II")], atol=1e-6)


def test_default_is_100hz_and_bad_fs_rejected(ptb):
    assert FS == 100 and N_SAMPLES == 1000
    X, _, _ = load_ptbxl(ptb, [3], cache_dir=None)
    assert X.shape[1] == 1000
    with pytest.raises(ValueError):
        load_ptbxl(ptb, [3], cache_dir=None, fs=250)


def test_fold_selection(ptb):
    for folds in ([9], [10], [1, 2, 3, 4, 5, 6, 7, 8]):
        X, Y, meta = load_ptbxl(ptb, folds, cache_dir=None)
        ids, _ = _expected(ptb, folds)
        assert set(meta.strat_fold) == set(folds)
        assert len(X) == len(Y) == len(meta) == len(ids)
    _, _, tr = load_ptbxl(ptb, list(range(1, 9)), cache_dir=None)
    _, _, te = load_ptbxl(ptb, [10], cache_dir=None)
    assert not set(tr.patient_id) & set(te.patient_id)
    with pytest.raises(ValueError):
        load_ptbxl(ptb, [11], cache_dir=None)


def test_cache_round_trip(ptb, tmp_path, monkeypatch):
    X1, Y1, m1 = load_ptbxl(ptb, [1, 2], cache_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*.npz"))) == 1

    def boom(*a, **k):
        raise AssertionError("records re-read despite cache")
    monkeypatch.setattr(data.wfdb, "rdrecord", boom)
    X2, Y2, m2 = load_ptbxl(ptb, [2, 1], cache_dir=str(tmp_path))   # order-insensitive key
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(Y1, Y2)
    pd.testing.assert_frame_equal(m1, m2)
    assert X2.dtype == np.float32
    with pytest.raises(AssertionError):                               # different key -> miss
        load_ptbxl(ptb, [3], cache_dir=str(tmp_path))
    with pytest.raises(AssertionError):                               # fs is part of the key
        load_ptbxl(ptb, [1, 2], cache_dir=str(tmp_path), fs=500)


# --------------------------------------------------------------------------- preprocess
def test_preprocess_zscore():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, N_SAMPLES)).astype(np.float32) * 3 + 5
    y = preprocess(x)
    assert y.shape == x.shape and y.dtype == np.float32
    np.testing.assert_allclose(y.mean(-1), 0, atol=1e-5)
    np.testing.assert_allclose(y.std(-1), 1, atol=1e-4)
    np.testing.assert_allclose(preprocess(x[1]), y[1], atol=1e-6)     # per-record


def test_preprocess_attenuates_50hz_at_500hz():
    fs, n = 500, 5000
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
    spec = np.abs(np.fft.rfft(preprocess(x, fs)))
    f = np.fft.rfftfreq(n, 1 / fs)
    ratio_db = 20 * np.log10(spec[f == 50][0] / spec[f == 10][0])
    assert ratio_db < -15.0            # zero-phase 4th-order design gives ~ -17.9 dB at 50 Hz


def test_preprocess_band_at_100hz():
    t = np.arange(N_SAMPLES) / FS
    x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 47 * t) + np.sin(2 * np.pi * 0.1 * t)
    spec = np.abs(np.fft.rfft(preprocess(x)))
    f = np.fft.rfftfreq(N_SAMPLES, 1 / FS)
    db = lambda f0: 20 * np.log10(spec[np.argmin(abs(f - f0))] / spec[f == 10][0])  # noqa: E731
    assert db(47) < -10.0 and db(0.1) < -20.0


# --------------------------------------------------------------------------- NSTDB
def test_nstdb_split(syn):
    root = str(syn / "nstdb")
    ns = load_nstdb(root, target_fs=500)
    assert set(ns) == {"bw", "ma", "em"}
    for name, parts in ns.items():
        raw = wfdb.rdrecord(os.path.join(root, name)).p_signal
        cut = int(round(0.6 * len(raw)))
        for part, seg in (("train", raw[:cut]), ("test", raw[cut:])):
            arr = parts[part]
            assert arr.dtype == np.float32 and arr.shape[1] == 2
            assert arr.shape[0] == int(np.ceil(len(seg) * 500 / 360))         # 360 -> 500 Hz
            np.testing.assert_allclose(arr, resample_poly(seg, 25, 18, axis=0), atol=1e-5)
        n_tr, n_te = len(parts["train"]), len(parts["test"])
        assert abs(n_tr / (n_tr + n_te) - 0.6) < 1e-3
        # The equalities above show train is a function of raw[:cut] only and test of
        # raw[cut:] only (each half resampled on its own), i.e. disjoint in time.


# --------------------------------------------------------------------------- Dataset
@pytest.fixture(scope="module")
def ds_parts(syn, ptb):
    X, Y, _ = load_ptbxl(ptb, [1, 2, 3], cache_dir=None)
    return X, Y, Corruptor(None, "train")  # Phase 1: 100 Hz, synthetic families only


def test_dataset_unpaired(ds_parts):
    X, Y, cor = ds_parts
    clean = ECGDataset(X, Y)
    s = clean[3]
    assert set(s) == {"x", "y", "idx"}
    assert s["x"].shape == (1, N_SAMPLES) and s["x"].dtype == torch.float32
    assert s["y"].shape == (5,) and s["y"].dtype == torch.float32 and s["idx"] == 3
    np.testing.assert_allclose(s["x"][0].numpy(), preprocess(X[3]), atol=1e-6)
    assert torch.equal(ECGDataset(X, Y, cor, p_corrupt=0.0)[3]["x"], s["x"])
    assert not torch.equal(ECGDataset(X, Y, cor, p_corrupt=1.0)[3]["x"], s["x"])
    b = next(iter(torch.utils.data.DataLoader(ECGDataset(X, Y, cor), batch_size=8)))
    assert b["x"].shape == (8, 1, N_SAMPLES) and b["y"].shape == (8, 5)


def test_dataset_paired(ds_parts):
    X, Y, cor = ds_parts
    ds = ECGDataset(X, Y, cor, paired=True, seed=1)
    for i in range(10):
        s = ds[i]
        assert set(s) == {"x_clean", "x_corr", "y", "idx", "severity"}
        assert s["x_clean"].shape == s["x_corr"].shape == (1, N_SAMPLES)
        assert s["x_clean"].dtype == s["x_corr"].dtype == torch.float32
        assert s["severity"] in (0, 1, 2)
        np.testing.assert_allclose(s["x_clean"][0].numpy(), preprocess(X[i]), atol=1e-6)
        assert not torch.equal(s["x_clean"], s["x_corr"])
        np.testing.assert_array_equal(s["y"].numpy(), Y[i])
    with pytest.raises(ValueError):
        ECGDataset(X, Y, None, paired=True)


def test_dataset_seed_determinism(ds_parts):
    X, Y, cor = ds_parts
    a = [ECGDataset(X, Y, cor, paired=True, seed=5)[i]["x_corr"] for i in range(3)]
    b = [ECGDataset(X, Y, cor, paired=True, seed=5)[i]["x_corr"] for i in range(3)]
    c = [ECGDataset(X, Y, cor, paired=True, seed=6)[i]["x_corr"] for i in range(3)]
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert not all(torch.equal(u, v) for u, v in zip(a, c))


def test_worker_rng_reproducible_and_epoch_varying(ds_parts):
    X, Y, cor = ds_parts
    ds = ECGDataset(X[:8], Y[:8], cor, paired=True, seed=0)

    def epoch(loader):
        return torch.cat([b["x_corr"] for b in loader])

    def run():
        torch.manual_seed(123)
        dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2, persistent_workers=True,
                                         worker_init_fn=data.worker_init_fn)
        return epoch(dl), epoch(dl)
    e1, e2 = run()
    r1, _ = run()
    assert torch.equal(e1, r1)             # reproducible given torch seed
    assert not torch.equal(e1, e2)         # new corruptions each epoch
