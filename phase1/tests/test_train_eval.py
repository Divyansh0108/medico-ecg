"""Unit tests for train.py / eval.py logic. Uses local fakes of data/corruptions/models
(injected via sys.modules) so these tests do not depend on the other modules."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval as ev  # noqa: E402
import train as tr  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "configs" / "phase1.yaml"


# --------------------------------------------------------------------------- variants / config
def test_variant_mapping():
    exp = {"A": ("cnn", False, False, False), "B": ("tcn", False, False, False),
           "C": ("concat", False, False, False), "D": ("gate", False, False, False),
           "E": ("racer", False, False, False), "F": ("racer", True, True, True),
           "G": ("cnn", True, False, True)}
    for v, (m, paired, rank, cons) in exp.items():
        s = tr.VARIANTS[v]
        assert (s.model, s.paired, s.rank, s.consistency) == (m, paired, rank, cons)


@pytest.mark.parametrize("v", ["F", "G"])
def test_paired_variants_rejected_under_clean(v):
    with pytest.raises(ValueError):
        tr.get_variant(v, "clean")
    assert tr.get_variant(v, "aug").paired
    with pytest.raises(SystemExit):  # CLI rejects too
        tr.main(["--variant", v, "--regime", "clean", "--seed", "0"])


def test_unknown_variant_and_regime():
    with pytest.raises(ValueError):
        tr.get_variant("H", "aug")
    with pytest.raises(ValueError):
        tr.get_variant("A", "noisy")


def test_config_smoke_override():
    full, smoke = tr.load_config(CFG), tr.load_config(CFG, smoke=True)
    assert full["train"]["lr"] == 1e-3 and full["train"]["batch_size"] == 64
    assert full["train"]["max_epochs"] == 40 and full["train"]["seeds"] == [0, 1, 2, 3, 4]
    assert smoke["train"]["max_epochs"] < full["train"]["max_epochs"]
    assert smoke["train"]["lr"] == full["train"]["lr"]  # untouched keys survive the merge
    assert smoke["paths"]["runs_dir"] != full["paths"]["runs_dir"]
    assert "smoke" in full and full["smoke"] is False and smoke["smoke"] is True


def test_config_phase1_scope():
    cfg = tr.load_config(CFG)
    assert cfg["data"]["fs"] == 100 and cfg["data"]["lead"] == "II"
    assert cfg["data"]["train_folds"] == list(range(1, 9))
    assert cfg["data"]["val_folds"] == [9] and cfg["data"]["test_folds"] == [10]
    assert cfg["variants"] == ["A", "C", "E"]
    assert cfg["corruptions"]["train_families"] == ["baseline_wander", "emg"]
    assert cfg["corruptions"]["unseen_families"] == ["motion_burst", "dropout"]
    assert not tr.needs_nstdb(cfg)
    assert cfg["eval"]["binding_groups"] == ["mixed"]


def test_sweep_jobs():
    cfg = tr.load_config(CFG)
    assert tr.sweep_jobs(cfg) == [("A", "clean"), ("A", "aug"), ("C", "clean"), ("C", "aug"),
                                  ("E", "clean"), ("E", "aug")]
    cfg["variants"] = ["E", "F", "G"]
    assert tr.sweep_jobs(cfg) == [("E", "clean"), ("E", "aug"), ("F", "aug"), ("G", "aug")]


# --------------------------------------------------------------------------- losses
def test_rank_loss_zero_when_margin_satisfied():
    r_clean = torch.full((4, 10), 0.9)
    r_corr = torch.full((4, 10), 0.7)          # 0.7 < 0.9 - 0.1
    assert tr.rank_loss(r_clean, r_corr, 0.1).item() == 0.0
    r_corr2 = torch.full((4, 10), 0.85)        # violates by 0.05
    assert tr.rank_loss(r_clean, r_corr2, 0.1).item() == pytest.approx(0.05, abs=1e-6)


def test_rank_loss_uses_per_sample_time_mean():
    r_clean = torch.tensor([[1.0, 0.0], [1.0, 1.0]])  # means 0.5, 1.0
    r_corr = torch.tensor([[0.5, 0.5], [0.0, 0.0]])   # means 0.5, 0.0
    # sample 0: relu(0.5-0.5+0.1)=0.1, sample 1: relu(0-1+0.1)=0 -> mean 0.05
    assert tr.rank_loss(r_clean, r_corr, 0.1).item() == pytest.approx(0.05)


def test_consistency_only_on_mild_pairs():
    lc = torch.zeros(3, 5, requires_grad=True)
    lx = torch.tensor([[0.0] * 5, [5.0] * 5, [5.0] * 5], requires_grad=True)
    # only row 0 is mild and it matches exactly -> 0, despite big severe/moderate gaps
    assert tr.consistency_loss(lc, lx, torch.tensor([0, 1, 2])).item() == 0.0
    # no mild pairs -> exactly zero, still differentiable
    z = tr.consistency_loss(lc, lx, torch.tensor([1, 2, 2]))
    assert z.item() == 0.0
    z.backward()
    # mild row with a gap -> MSE of sigmoids over mild rows only, gradient to both views
    lc2 = torch.zeros(3, 5, requires_grad=True)
    lx2 = lx.detach().clone().requires_grad_(True)
    loss = tr.consistency_loss(lc2, lx2, torch.tensor([1, 0, 2]))
    assert loss.item() == pytest.approx((torch.sigmoid(torch.tensor(5.0)) - 0.5).item() ** 2, rel=1e-5)
    loss.backward()
    assert lc2.grad[1].abs().sum() > 0 and lx2.grad[1].abs().sum() > 0
    assert lc2.grad[[0, 2]].abs().sum() == 0 and lx2.grad[[0, 2]].abs().sum() == 0


class _TinyGate(nn.Module):
    def __init__(self, with_r=True):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, 9, stride=8)
        self.head = nn.Linear(4, 5)
        self.with_r = with_r

    def forward(self, x):
        f = self.conv(x)
        r_map = torch.sigmoid(f) if self.with_r else None
        logits = self.head(f.mean(-1))
        return {"logits": logits, "r": None if r_map is None else r_map.mean(1), "r_map": r_map}


def test_compute_loss_paired_composition():
    torch.manual_seed(0)
    m = _TinyGate()
    b = 6
    batch = {"x_clean": torch.randn(b, 1, 1000), "x_corr": torch.randn(b, 1, 1000),
             "y": torch.randint(0, 2, (b, 5)).float(), "severity": torch.tensor([0, 1, 2, 0, 1, 2])}
    T = {"lambda1": 0.1, "lambda2": 0.1, "margin": 0.1}
    loss, parts = tr.compute_loss(m, batch, tr.VARIANTS["F"], T, torch.device("cpu"), None)
    assert set(parts) == {"bce", "rank", "consistency"}
    assert loss.item() == pytest.approx(parts["bce"] + 0.1 * parts["rank"] + 0.1 * parts["consistency"], rel=1e-5)
    loss_g, parts_g = tr.compute_loss(_TinyGate(with_r=False), batch, tr.VARIANTS["G"], T, torch.device("cpu"), None)
    assert set(parts_g) == {"bce", "consistency"}


# --------------------------------------------------------------------------- metrics
def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (300, 5))
    p = np.round(rng.random((300, 5)) + 0.3 * y, 1)  # ties on purpose
    ours = tr.auroc_per_class(y, p)
    ref = [roc_auc_score(y[:, k], p[:, k]) for k in range(5)]
    np.testing.assert_allclose(ours, ref, atol=1e-12)
    y[:, 2] = 0
    assert np.isnan(tr.auroc_per_class(y, p)[2])
    assert np.isfinite(tr.macro_auroc(y, p))


def test_threshold_tuning_is_f1_optimal():
    y = np.array([[0], [0], [1], [1], [0], [1]])
    p = np.array([[0.1], [0.2], [0.3], [0.8], [0.9], [0.95]])
    th = tr.tune_thresholds(y, p)
    # brute force over candidate thresholds
    best = max(np.unique(p), key=lambda t: (tr.macro_f1(y, p, np.array([t])), -t))
    assert tr.macro_f1(y, p, th) == pytest.approx(tr.macro_f1(y, p, np.array([best])))
    assert th[0] == pytest.approx(0.3)
    # perfectly separable -> F1 == 1; class without positives -> default 0.5
    y2 = np.array([[0, 0], [1, 0], [1, 0]])
    p2 = np.array([[0.2, 0.1], [0.6, 0.3], [0.7, 0.9]])
    th2 = tr.tune_thresholds(y2, p2)
    assert th2[1] == 0.5 and tr.macro_f1(y2[:, :1], p2[:, :1], th2[:1]) == 1.0


# --------------------------------------------------------------------------- bootstrap
def test_bootstrap_resamples_whole_patients():
    pids = np.array([7, 7, 7, 3, 9, 9, 1, 1, 1, 1])
    sizes = {p: (pids == p).sum() for p in np.unique(pids)}
    idxs = ev.patient_bootstrap_indices(pids, 50, np.random.default_rng(0))
    for idx in idxs:
        drawn = pids[idx]
        for p in np.unique(drawn):
            assert (drawn == p).sum() % sizes[p] == 0  # every draw brings the full cluster
        assert len(np.unique(idx[np.isin(pids[idx], [7])])) in (0, 3)
    # number of clusters drawn equals number of patients
    for idx in idxs[:10]:
        n_clusters = sum((pids[idx] == p).sum() // sizes[p] for p in sizes)
        assert n_clusters == len(sizes)


def test_bootstrap_ci_contains_true_difference():
    rng = np.random.default_rng(1)
    n_pat, per = 400, 3
    pids = np.repeat(np.arange(n_pat), per)
    y = rng.integers(0, 2, (n_pat * per, 2))
    # model a separates better than model b
    pa = y + rng.normal(0, 0.8, y.shape)
    pb = y + rng.normal(0, 1.6, y.shape)
    probs = {"a": {"c": pa}, "b": {"c": pb}}
    point = {m: {"c": tr.macro_auroc(y, probs[m]["c"])} for m in probs}
    idx = ev.patient_bootstrap_indices(pids, 300, np.random.default_rng(2))
    boot = ev.bootstrap_auroc(y, probs, idx)
    d, lo, hi = ev.diff_ci(point["a"], point["b"], boot["a"], boot["b"], ["c"])
    # population difference via a huge independent sample of the same generative model
    yb = rng.integers(0, 2, (200000, 2))
    true = tr.macro_auroc(yb, yb + rng.normal(0, 0.8, yb.shape)) - tr.macro_auroc(yb, yb + rng.normal(0, 1.6, yb.shape))
    assert lo < true < hi and lo < d < hi and lo > 0
    # identical models -> CI is exactly [0, 0]
    d0, lo0, hi0 = ev.diff_ci(point["a"], point["a"], boot["a"], boot["a"], ["c"])
    assert d0 == lo0 == hi0 == 0.0


# --------------------------------------------------------------------------- decision rule
def _cand(ci_lo=0.01, r=(0.9, 0.8, 0.6), r_abn=0.88, r_norm=0.9):
    return {"ci": {g: ("C", 0.02, ci_lo, 0.04) for g in ev.BINDING_GROUPS},
            "supporting": {"mixed (severe)": ("C", 0.03, -0.01, 0.06)},
            "r_sev": dict(zip(ev.SEV_ORDER, r)), "r_abn": r_abn, "r_norm": r_norm}


def test_decision_go():
    d = ev.apply_decision_rule({"E": _cand(), "F": _cand(ci_lo=-0.01)})
    assert d["verdict"] == "GO" and d["passing"] == ["E"]
    # supporting (non-binding) failures do not affect the verdict
    assert any(not r["binding"] and not r["passed"] for r in d["criteria"])


@pytest.mark.parametrize("kw", [
    {"ci_lo": -0.001},                    # CI includes 0 -> baselines match
    {"r": (0.9, 0.9, 0.6)},               # r not strictly monotone
    {"r": (0.6, 0.8, 0.9)},               # r increases with severity
    {"r_abn": 0.84, "r_norm": 0.9},       # r drops on clean-abnormal by > 0.05
])
def test_decision_nogo_paths(kw):
    d = ev.apply_decision_rule({"E": _cand(**kw), "F": _cand(**kw)})
    assert d["verdict"] == "NO-GO / PIVOT" and d["passing"] == []


def test_decision_tolerance_boundary_and_one_group_failing():
    assert ev.apply_decision_rule({"E": _cand(r_abn=0.85, r_norm=0.9)})["verdict"] == "GO"  # exactly -0.05
    c = _cand()
    c["ci"]["real_noise"] = ("B", 0.01, -0.002, 0.02)  # passes mixed, fails real-noise
    assert ev.apply_decision_rule({"E": c})["verdict"] == "GO"      # real_noise not binding in Phase 1
    assert ev.apply_decision_rule({"E": c}, binding_groups=["mixed", "real_noise"])["verdict"] == "NO-GO / PIVOT"


def test_decision_incomplete():
    assert ev.apply_decision_rule({})["verdict"] == "INCOMPLETE"
    c = _cand()
    c["r_sev"] = None
    assert ev.apply_decision_rule({"E": c})["verdict"] == "INCOMPLETE"


# --------------------------------------------------------------------------- conditions
def test_conditions_and_groups():
    fams = ["nstdb_bw", "nstdb_ma", "baseline_wander", "emg", "nstdb_em", "motion_burst", "powerline", "dropout"]
    conds = ev.build_conditions(fams, ev.SEV_ORDER, ["whole", "burst"])
    names = [c.name for c in conds]
    assert names[0] == "clean" and len(set(names)) == len(names)
    assert len(conds) == 1 + 7 * 3 * 2 + 3 + 3 * 2  # dropout once per severity; mixed x sev x mode
    assert "dropout" not in ev.mix_pool(fams)
    g = ev.group_families(fams[:4], fams[4:])
    assert "nstdb_em" not in g["unseen_no_em"] and "nstdb_em" in g["unseen_families"]
    assert len(ev.group_members(conds, g["real_noise"], "severe")) == 6


def test_conditions_and_groups_phase1():
    tf, uf = ["baseline_wander", "emg"], ["motion_burst", "dropout"]
    conds = ev.build_conditions(tf + uf, ev.SEV_ORDER, ["whole", "burst"])
    assert len(conds) == 1 + 3 * 3 * 2 + 3 + 3 * 2
    assert ev.mix_pool(tf + uf) == ["baseline_wander", "emg", "motion_burst"]
    g = ev.group_families(tf, uf)
    assert set(g) == {"seen_families", "unseen_families", "mixed"}   # no real_noise / unseen_no_em


# --------------------------------------------------------------------------- end-to-end with fakes
def _install_fakes(monkeypatch, n_per_fold=24):
    FAMS_T = ["baseline_wander", "emg"]
    FAMS_U = ["motion_burst", "dropout"]

    def load_ptbxl(root, folds, lead="II", cache_dir=None, fs=100):
        assert fs == 100
        rng = np.random.default_rng(sum(folds))
        n = n_per_fold * len(folds)
        Y = np.zeros((n, 5), np.float32)
        Y[np.arange(n), np.arange(n) % 5] = 1
        t = np.arange(1000) / 100
        X = (np.sin(2 * np.pi * (1 + Y.argmax(1))[:, None] * t) + 0.3 * rng.normal(size=(n, 1000))).astype(np.float32)
        meta = pd.DataFrame({"ecg_id": np.arange(n) + 1000 * folds[0], "patient_id": np.arange(n) // 2,
                             "strat_fold": np.repeat(folds, n_per_fold)})
        return X, Y, meta

    def preprocess(x, fs=100):
        x = np.asarray(x, np.float32)
        return ((x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)).astype(np.float32)

    class Corruptor:
        def __init__(self, nstdb, split, fs=100, train_families=None):
            assert split in ("train", "test") and nstdb is None and fs == 100

        def apply(self, x, family, severity, mode, rng):
            snr = {"mild": 15.0, "moderate": 6.0, "severe": 0.0}[severity]
            noise = rng.normal(size=x.shape)
            return x + noise * np.sqrt(x.var() / 10 ** (snr / 10))

        def sample_train(self, x, rng):
            s = int(rng.integers(3))
            return self.apply(x, "emg", ["mild", "moderate", "severe"][s], "whole", rng), s

    class ECGDataset(torch.utils.data.Dataset):
        def __init__(self, X_raw, Y, corruptor=None, p_corrupt=0.5, paired=False, seed=0, fs=100):
            self.X, self.Y, self.c, self.p, self.paired = X_raw, Y, corruptor, p_corrupt, paired
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            x = self.X[i]
            t = lambda a: torch.from_numpy(preprocess(a))[None]  # noqa: E731
            if self.paired:
                xc, s = self.c.sample_train(x, self.rng)
                return {"x_clean": t(x), "x_corr": t(xc), "y": torch.from_numpy(self.Y[i]), "idx": i, "severity": s}
            if self.c is not None and self.rng.random() < self.p:
                x, _ = self.c.sample_train(x, self.rng)
            return {"x": t(x), "y": torch.from_numpy(self.Y[i]), "idx": i}

    data = types.ModuleType("data")
    data.load_ptbxl, data.preprocess, data.ECGDataset = load_ptbxl, preprocess, ECGDataset
    def load_nstdb(*a, **k):
        raise AssertionError("Phase 1 must not read NSTDB")
    data.load_nstdb = load_nstdb
    data.worker_init_fn = lambda wid: None
    corr = types.ModuleType("corruptions")
    corr.Corruptor, corr.TRAIN_FAMILIES, corr.UNSEEN_FAMILIES = Corruptor, FAMS_T, FAMS_U
    corr.ALL_FAMILIES, corr.MODES, corr.SEVERITIES = FAMS_T + FAMS_U, ["whole", "burst"], ["mild", "moderate", "severe"]
    models = types.ModuleType("models")
    models.build_model = lambda name, n_classes=5, **kw: _TinyGate(with_r=name in ("gate", "racer"))
    models.count_params = lambda m: sum(p.numel() for p in m.parameters())
    for name, mod in (("data", data), ("corruptions", corr), ("models", models)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_end_to_end_with_fakes(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    monkeypatch.setattr(tr, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(ev, "get_device", lambda: torch.device("cpu"))
    cfg = tr.load_config(CFG, smoke=True)
    cfg["paths"] = {k: str(tmp_path / k) for k in cfg["paths"]}
    cfg["eval"]["n_bootstrap"] = 10
    for v, reg in tr.sweep_jobs(cfg):   # Phase 1: A, C, E x {clean, aug}
        for s in (0, 1):
            d = tr.train(cfg, v, reg, s)
            meta = json.loads((d / "meta.json").read_text())
            assert meta["amp_dtype"] == "float32" and meta["git_hash"] and meta["fs"] == 100
            assert set(json.loads((d / "thresholds.json").read_text())) == set(tr.SUPERCLASSES)
            assert list(pd.read_csv(d / "log.csv").columns[:3]) == ["epoch", "train_loss", "val_macro_auroc"]
    # resumable: a complete run is skipped (best.pt untouched)
    d = Path(cfg["paths"]["runs_dir"]) / "A_aug_s0"
    mtime = (d / "best.pt").stat().st_mtime_ns
    tr.train(cfg, "A", "aug", 0)
    assert (d / "best.pt").stat().st_mtime_ns == mtime

    dec = ev.evaluate(cfg, Path(cfg["paths"]["runs_dir"]))
    out = Path(cfg["paths"]["results_dir"])
    assert dec["verdict"] in ("GO", "NO-GO / PIVOT", "INCOMPLETE")
    pr = pd.read_csv(out / "per_run.csv")
    assert list(pr.columns) == ["variant", "regime", "seed", "condition", "family", "severity", "mode",
                                "macro_auroc", "macro_f1"]
    assert pr.condition.nunique() == 1 + 3 * 6 + 3 + 6
    assert set(zip(pr.variant, pr.regime)) == set(tr.sweep_jobs(cfg))
    rel = pd.read_csv(out / "reliability.csv")
    assert set(rel.variant) == {"E"}
    assert {"clean_norm", "clean_abnormal"} <= set(rel.subset)
    b = pd.read_csv(out / "bootstrap.csv")
    assert set(b.candidate) == {"E"} and set(b.baseline) == {"A", "C"}
    assert "real_noise" not in set(b.group)
    summary = (out / "summary.md").read_text()
    assert summary.count("Verdict") == 1
    assert "real_noise" not in summary and "unseen_no_em" not in summary
    assert "No real (NSTDB) noise" in summary and "at 100 Hz" in summary
    assert "on mixed" in summary and "Baselines: augmented A, C." in summary
    assert (out / "figures" / "auroc_vs_snr_aug.png").exists() and (out / "figures" / "r_vs_snr.png").exists()
    # the corrupted test set is cached and deterministic: a second evaluation reproduces per_run.csv
    ev.evaluate(cfg, Path(cfg["paths"]["runs_dir"]))
    pd.testing.assert_frame_equal(pr, pd.read_csv(out / "per_run.csv"))
