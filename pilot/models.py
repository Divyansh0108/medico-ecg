"""Model zoo for the RACER Phase-1 experiment (docs/master.md Part B4, docs/CONTRACT.md).

All models map a single-lead, preprocessed 10 s ECG ``x: [B, 1, 10*fs]`` to a dict
``{"logits": [B, n_classes], "r": [B, T'] | None, "r_map": [B, C, T'] | None}``.
Phase 1 uses fs = 100 Hz (x: [B, 1, 1000]); fs = 500 is kept for later phases.
``fs`` only changes the stem strides and the local-branch depth (``ARCH``), chosen
so receptive fields in seconds stay comparable across rates.

Building blocks (shared by every model)
---------------------------------------
* ``Stem``: three Conv-BN-ReLU layers (k=7/5/5, channels 32 -> 64 -> C).
  100 Hz: strides (1,1,2), total stride 2, T' = 500 at 50 Hz (20 ms/step), RF 15.
  500 Hz: strides (2,2,2), total stride 8, T' = 625 at 62.5 Hz (16 ms/step), RF 31.
* ``ResBlock``: Conv(k, d)-BN-ReLU-Conv(k, d)-BN + identity (or 1x1-conv-BN
  shortcut when stride/width change), then ReLU. Non-causal "same" padding
  ``d*(k-1)//2``. BatchNorm everywhere (batch 64 is large enough; no reason for
  GroupNorm).

Architectures
-------------
* ``cnn``   (A): Stem(C=128) + 5 ResBlocks, k=7, stride (1,2,1,2,1), GAP, linear.
* ``tcn``   (B): Stem(C=168) + 7 non-causal dilated ResBlocks, k=3,
  d = 1,2,...,64, GAP, linear.
* Two-branch models share the exact same two branches (each with its OWN stem,
  no weight sharing, so the local branch cannot inherit context features):
    - local  : Stem(C=128) + ResBlocks k=3, d=1 (3 blocks @100 Hz, 4 @500 Hz)
    - context: Stem(C=128) + 7 ResBlocks, k=3, d=1..64
  Both output time-aligned maps F_local, F_ctx of shape [B, 128, T'].
* ``concat`` (C): GAP(cat[F_local, F_ctx]) -> linear(256 -> n_classes).
* ``gate``   (D): g = sigmoid(MLP([F_local, F_ctx])), F = g*F_local + (1-g)*F_ctx.
* ``racer``  (E): r = sigmoid(MLP([F_local, F_ctx, |F_local - P(F_ctx)|])),
  F_cal = r*F_local + (1-r)*F_ctx, P = 1x1 conv (C -> C, with bias).
  The gate MLP is two 1x1 convs (in -> 128, ReLU, 128 -> C). D and E are
  identical except for the extra |F_local - P(F_ctx)| input (hence P and the
  wider first MLP layer: +32,896 parameters for E vs D).
  The gate is CHANNEL-WISE: r_map has shape [B, C, T'] (one weight per channel
  per timestep); the reported reliability r = r_map.mean(dim=1), shape [B, T'].
  The last gate-MLP bias is 0 at init, so r starts near 0.5 (no initial bias
  toward either branch).

Parameter counts / receptive fields (n_classes=5; verified by
``tests/test_models.py`` and ``python models.py``)::

    100 Hz (Phase 1)
    model   params     T'   RF (samples / seconds)                  gate
    cnn     1,235,237  125  291 / 2.91 s                            -
    tcn     1,255,709  500  1031 / 10.31 s                          -
    concat  1,093,189  500  local 39 / 0.39 s ; ctx 1031 / 10.31 s  -
    gate    1,141,957  500  local 39 / 0.39 s ; ctx 1031 / 10.31 s  [B,C,T']
    racer   1,174,853  500  local 39 / 0.39 s ; ctx 1031 / 10.31 s  [B,C,T']

    500 Hz
    cnn     1,235,237  157  1135 / 2.27 s                           -
    tcn     1,255,709  625  4095 / 8.19 s                           -
    concat  1,192,005  625  local 159 / 0.32 s ; ctx 4095 / 8.19 s  -
    gate    1,240,773  625  local 159 / 0.32 s ; ctx 4095 / 8.19 s  [B,C,T']
    racer   1,273,669  625  local 159 / 0.32 s ; ctx 4095 / 8.19 s  [B,C,T']

RFs are computed analytically by ``receptive_field`` (main path of each residual
block; shortcuts never enlarge the RF) and cross-checked empirically in the tests.
"""
from __future__ import annotations

import torch
import torch.nn as nn

MODEL_NAMES = ["cnn", "tcn", "concat", "gate", "racer"]

C_BRANCH = 128      # channels of F_local / F_ctx (identical by construction)
GATE_HIDDEN = 128   # hidden width of the 1x1-conv gate MLP


# --------------------------------------------------------------------------- blocks
class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, k: int, stride: int = 1):
        super().__init__(
            nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )
        self.rf_layers = [(k, stride, 1)]


class Stem(nn.Sequential):
    """Strided stem. 500 Hz: strides (2,2,2), 5000 -> 625; 100 Hz: (1,1,2), 1000 -> 500."""

    def __init__(self, c_out: int, strides=(2, 2, 2)):
        s1, s2, s3 = strides
        super().__init__(
            ConvBNReLU(1, 32, 7, s1),
            ConvBNReLU(32, 64, 5, s2),
            ConvBNReLU(64, c_out, 5, s3),
        )
        self.rf_layers = [l for m in self for l in m.rf_layers]


class ResBlock(nn.Module):
    """Non-causal residual block: two (dilated) convs + identity/1x1 shortcut."""

    def __init__(self, c_in: int, c_out: int, k: int, dilation: int = 1, stride: int = 1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Identity() if stride == 1 and c_in == c_out
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))
        )
        self.rf_layers = [(k, stride, dilation), (k, 1, dilation)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.shortcut(x))


def receptive_field(branch: nn.Sequential) -> int:
    """Analytic RF (input samples) of a chain of Stem/ConvBNReLU/ResBlock modules."""
    rf, jump = 1, 1
    for m in branch:
        for k, s, d in m.rf_layers:
            rf += (k - 1) * d * jump
            jump *= s
    return rf


# Per input rate: stem strides and local-branch depth, chosen so the receptive fields in
# SECONDS stay close across rates (local ~0.3-0.4 s = one complex; context ~8-10 s).
ARCH = {500: {"stem": (2, 2, 2), "local_blocks": 4},
        100: {"stem": (1, 1, 2), "local_blocks": 3}}
FS_DEFAULT = 100


def _arch(fs: int) -> dict:
    if fs not in ARCH:
        raise ValueError(f"fs must be one of {sorted(ARCH)}, got {fs}")
    return ARCH[fs]


def local_branch(c: int = C_BRANCH, n_blocks: int = 4, stem=(2, 2, 2)) -> nn.Sequential:
    """Shallow CNN, small RF (0.39 s @100 Hz / 0.32 s @500 Hz): one wave/complex."""
    return nn.Sequential(Stem(c, stem), *[ResBlock(c, c, 3) for _ in range(n_blocks)])


def context_branch(c: int = C_BRANCH, n_blocks: int = 7, stem=(2, 2, 2)) -> nn.Sequential:
    """Non-causal dilated TCN, dilation 2^i, large RF (10.3 s @100 Hz / 8.19 s @500 Hz)."""
    return nn.Sequential(Stem(c, stem), *[ResBlock(c, c, 3, dilation=2 ** i) for i in range(n_blocks)])


def _out(logits, r_map=None):
    r = None if r_map is None else r_map.mean(dim=1)
    return {"logits": logits, "r": r, "r_map": r_map}


# --------------------------------------------------------------------------- single-branch
class CNN(nn.Module):
    """A: 1D ResNet, 5 blocks (k=7), stride 2 in blocks 2 and 4."""

    def __init__(self, n_classes: int = 5, c: int = 128, k: int = 7, strides=(1, 2, 1, 2, 1),
                 fs: int = FS_DEFAULT):
        super().__init__()
        self.encoder = nn.Sequential(Stem(c, _arch(fs)["stem"]), *[ResBlock(c, c, k, stride=s) for s in strides])
        self.head = nn.Linear(c, n_classes)

    def forward(self, x):
        return _out(self.head(self.encoder(x).mean(dim=-1)))


class TCN(nn.Module):
    """B: non-causal dilated TCN (same design as the context branch, wider)."""

    def __init__(self, n_classes: int = 5, c: int = 168, n_blocks: int = 7, fs: int = FS_DEFAULT):
        super().__init__()
        self.encoder = context_branch(c, n_blocks, _arch(fs)["stem"])
        self.head = nn.Linear(c, n_classes)

    def forward(self, x):
        return _out(self.head(self.encoder(x).mean(dim=-1)))


# --------------------------------------------------------------------------- two-branch
class TwoBranch(nn.Module):
    """Local CNN branch + context TCN branch -> time-aligned [B, C, T'] maps."""

    def __init__(self, c: int = C_BRANCH, fs: int = FS_DEFAULT):
        super().__init__()
        a = _arch(fs)
        self.local = local_branch(c, a["local_blocks"], a["stem"])
        self.context = context_branch(c, stem=a["stem"])

    def branches(self, x):
        f_local, f_ctx = self.local(x), self.context(x)
        assert f_local.shape == f_ctx.shape, (f_local.shape, f_ctx.shape)
        return f_local, f_ctx


class Concat(TwoBranch):
    """C: channel concatenation per timestep, GAP, linear head."""

    def __init__(self, n_classes: int = 5, c: int = C_BRANCH, fs: int = FS_DEFAULT):
        super().__init__(c, fs)
        self.head = nn.Linear(2 * c, n_classes)

    def forward(self, x):
        f_local, f_ctx = self.branches(x)
        return _out(self.head(torch.cat([f_local, f_ctx], dim=1).mean(dim=-1)))


class GatedFusion(TwoBranch):
    """D (disagreement=False) / E (disagreement=True): per-timestep channel-wise gate.

    gate input  D: [F_local, F_ctx]                         (2C channels)
                E: [F_local, F_ctx, |F_local - P(F_ctx)|]   (3C channels)
    r_map = sigmoid(Conv1x1(ReLU(Conv1x1(input))))          [B, C, T']
    F     = r_map * F_local + (1 - r_map) * F_ctx ; GAP ; linear head.
    """

    def __init__(self, n_classes: int = 5, c: int = C_BRANCH, hidden: int = GATE_HIDDEN,
                 disagreement: bool = False, fs: int = FS_DEFAULT):
        super().__init__(c, fs)
        self.proj = nn.Conv1d(c, c, 1) if disagreement else None   # P
        n_in = (3 if disagreement else 2) * c
        self.gate = nn.Sequential(nn.Conv1d(n_in, hidden, 1), nn.ReLU(inplace=True), nn.Conv1d(hidden, c, 1))
        nn.init.zeros_(self.gate[-1].bias)
        self.head = nn.Linear(c, n_classes)

    def gate_input(self, f_local, f_ctx):
        feats = [f_local, f_ctx]
        if self.proj is not None:
            feats.append((f_local - self.proj(f_ctx)).abs())
        return torch.cat(feats, dim=1)

    def forward(self, x):
        f_local, f_ctx = self.branches(x)
        r_map = torch.sigmoid(self.gate(self.gate_input(f_local, f_ctx)))
        fused = r_map * f_local + (1 - r_map) * f_ctx
        return _out(self.head(fused.mean(dim=-1)), r_map)


# --------------------------------------------------------------------------- API
def build_model(name: str, n_classes: int = 5, **kw) -> nn.Module:
    if name == "cnn":
        return CNN(n_classes, **kw)
    if name == "tcn":
        return TCN(n_classes, **kw)
    if name == "concat":
        return Concat(n_classes, **kw)
    if name == "gate":
        return GatedFusion(n_classes, disagreement=False, **kw)
    if name == "racer":
        return GatedFusion(n_classes, disagreement=True, **kw)
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def receptive_fields(model: nn.Module) -> dict[str, int]:
    """Analytic RF (samples) per encoder/branch of a built model."""
    if isinstance(model, TwoBranch):
        return {"local": receptive_field(model.local), "context": receptive_field(model.context)}
    return {"encoder": receptive_field(model.encoder)}


if __name__ == "__main__":
    for fs in sorted(ARCH):
        x = torch.zeros(2, 1, 10 * fs)
        print(f"fs={fs} Hz\n{'model':8s}{'params':>11s}{'T_out':>7s}  RF (samples)")
        for n in MODEL_NAMES:
            m = build_model(n, fs=fs).eval()
            with torch.no_grad():
                t = (m.local(x) if isinstance(m, TwoBranch) else m.encoder(x)).shape[-1]
            print(f"{n:8s}{count_params(m):>11,d}{t:>7d}  {receptive_fields(m)}")
