"""Tests for models.py (docs/CONTRACT.md: models.py section)."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import MODEL_NAMES, TwoBranch, build_model, count_params, receptive_fields  # noqa: E402

B, N_CLS = 3, 5
FSS = [100, 500]                      # Phase 1 runs at 100 Hz; 500 Hz kept for later phases
T_OUT = {100: 500, 500: 625}          # time steps of the two-branch / gate maps
GATED = {"gate", "racer"}
MPS = torch.backends.mps.is_available()
PARAMS_100 = {"cnn": 1_235_237, "tcn": 1_255_709, "concat": 1_093_189, "gate": 1_141_957, "racer": 1_174_853}


def _build(name, seed=0, fs=100):
    torch.manual_seed(seed)
    return build_model(name, n_classes=N_CLS, fs=fs)


@pytest.mark.parametrize("fs", FSS)
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_param_count_in_range(name, fs):
    assert 1_000_000 <= count_params(_build(name, fs=fs)) <= 3_000_000


def test_phase1_param_counts_and_rf_seconds():
    for name, n in PARAMS_100.items():
        assert count_params(_build(name)) == n
    rf = receptive_fields(_build("racer"))
    assert 0.3 <= rf["local"] / 100 <= 0.4       # one wave/complex
    assert rf["context"] / 100 >= 8.0             # most of the 10 s record
    with pytest.raises(ValueError):
        build_model("cnn", fs=250)


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        build_model("transformer")


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("fs", FSS)
def test_forward_contract_cpu(name, fs):
    m = _build(name, fs=fs).eval()
    with torch.no_grad():
        out = m(torch.randn(B, 1, 10 * fs))
    assert set(out) == {"logits", "r", "r_map"}
    assert out["logits"].shape == (B, N_CLS) and out["logits"].dtype == torch.float32
    assert torch.isfinite(out["logits"]).all()
    if name in GATED:
        r, r_map = out["r"], out["r_map"]
        assert r_map.dim() == 3 and r_map.shape[:1] == (B,) and r_map.shape[1] == 128
        assert r.shape == (B, r_map.shape[-1]) == (B, T_OUT[fs])
        assert torch.allclose(r, r_map.mean(dim=1))
        assert (r_map >= 0).all() and (r_map <= 1).all()
        assert (r >= 0).all() and (r <= 1).all()
    else:
        assert out["r"] is None and out["r_map"] is None


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("fs", FSS)
def test_determinism(name, fs):
    x = torch.randn(B, 1, 10 * fs, generator=torch.Generator().manual_seed(1))
    outs = []
    for _ in range(2):
        m = _build(name, seed=123, fs=fs).eval()
        with torch.no_grad():
            outs.append(m(x))
    assert torch.equal(outs[0]["logits"], outs[1]["logits"])
    if name in GATED:
        assert torch.equal(outs[0]["r_map"], outs[1]["r_map"])
    # different seed -> different weights
    m2 = _build(name, seed=124, fs=fs).eval()
    with torch.no_grad():
        assert not torch.equal(m2(x)["logits"], outs[0]["logits"])


@pytest.mark.parametrize("name", ["concat", "gate", "racer"])
@pytest.mark.parametrize("fs", FSS)
def test_branch_maps_aligned(name, fs):
    m = _build(name, fs=fs).eval()
    x = torch.randn(B, 1, 10 * fs)
    with torch.no_grad():
        f_local, f_ctx = m.branches(x)
    assert f_local.shape == f_ctx.shape == (B, 128, T_OUT[fs])


@pytest.mark.parametrize("fs", FSS)
def test_two_branch_models_share_architecture(fs):
    """C, D, E use byte-identical branch architectures (same param counts per branch)."""
    counts = {n: (count_params(_build(n, fs=fs).local), count_params(_build(n, fs=fs).context)) for n in ["concat", "gate", "racer"]}
    assert len(set(counts.values())) == 1


@pytest.mark.parametrize("fs", FSS)
def test_gate_vs_racer_differ_only_by_disagreement(fs):
    g, r = _build("gate", fs=fs), _build("racer", fs=fs)
    c, h = 128, 128
    # P (C*C + C) + extra C input channels in first gate conv (C*hidden)
    assert count_params(r) - count_params(g) == c * c + c + c * h
    assert g.proj is None and isinstance(r.proj, nn.Conv1d)
    assert g.gate[0].in_channels == 2 * c and r.gate[0].in_channels == 3 * c


@pytest.mark.parametrize("fs", FSS)
def test_racer_gate_depends_on_disagreement(fs):
    """r must receive gradient through |F_local - P(F_ctx)|: P is used nowhere else."""
    m = _build("racer", fs=fs).train()
    out = m(torch.randn(B, 1, 10 * fs))
    out["r"].sum().backward()
    assert m.proj.weight.grad is not None and m.proj.weight.grad.abs().sum() > 0
    # And zeroing the disagreement slice of the gate input changes r.
    m.eval()
    x = torch.randn(B, 1, 10 * fs)
    with torch.no_grad():
        r0 = m(x)["r_map"]
        m.gate[0].weight[:, 256:].zero_()
        r1 = m(x)["r_map"]
    assert not torch.allclose(r0, r1)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("fs", FSS)
def test_receptive_field_matches_empirical(name, fs):
    """Analytic RF == support of d(out[center])/d(input) with all-positive weights/input
    (no ReLU is ever inactive, so the gradient support is exactly the RF)."""
    m = _build(name, fs=fs).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.Conv1d):
                mod.weight.fill_(1.0 / (mod.in_channels * mod.kernel_size[0]))
    encoders = {"local": m.local, "context": m.context} if isinstance(m, TwoBranch) else {"encoder": m.encoder}
    analytic = receptive_fields(m)
    for key, enc in encoders.items():
        x = torch.ones(1, 1, 3 * 8192, requires_grad=True)   # long enough: no edge effects at centre
        y = enc(x)
        y[0, :, y.shape[-1] // 2].sum().backward()
        nz = (x.grad[0, 0] != 0).nonzero().squeeze(1)
        assert nz[-1] - nz[0] + 1 == analytic[key], (key, int(nz[-1] - nz[0] + 1), analytic[key])


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("fs", FSS)
def test_mps_autocast_fp16_forward_backward(name, fs):
    m = _build(name, fs=fs).to("mps").train()
    x = torch.randn(4, 1, 10 * fs, device="mps")
    y = torch.randint(0, 2, (4, N_CLS), device="mps").float()
    with torch.autocast("mps", dtype=torch.float16):
        out = m(x)
        loss = nn.functional.binary_cross_entropy_with_logits(out["logits"].float(), y)
        if name in GATED:
            loss = loss + out["r"].float().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert out["logits"].shape == (4, N_CLS)
    if name in GATED:
        assert out["r"].shape == (4, T_OUT[fs]) and out["r_map"].shape == (4, 128, T_OUT[fs])
        assert (out["r"] >= 0).all() and (out["r"] <= 1).all()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
