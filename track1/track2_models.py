"""Track 2 variants sharing one resnet1d_wang trunk (design fixed in results/track2/RULES.md, section D).

Split: shallow = stem + stage 1 (3 of 7 conv layers, 128 ch, full resolution); deep = stages 2-3.
F_local = Linear(GAP(shallow)) and F_ctx = Linear(GAP(deep)), both of size d. (A linear projection
followed by GAP equals GAP followed by the projection.)
  C: head([F_local, F_ctx])
  D: g = sigmoid(MLP([F_local, F_ctx])); F = g F_local + (1 - g) F_ctx
  E: r = sigmoid(MLP([F_local, F_ctx, |F_local - P F_ctx|])); F = r F_local + (1 - r) F_ctx
Gates are one scalar per window, kept in model.last_r after each forward (shape (B,)).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from strodthoff import ResNet1dWang


def _head(nin: int, nc: int, hidden: int = 128, ps: float = 0.5) -> nn.Sequential:
    return nn.Sequential(nn.BatchNorm1d(nin), nn.Dropout(ps / 2), nn.Linear(nin, hidden), nn.ReLU(inplace=True),
                         nn.BatchNorm1d(hidden), nn.Dropout(ps), nn.Linear(hidden, nc))


def _mlp(nin: int, h: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(nin, h), nn.ReLU(inplace=True), nn.Linear(h, 1))


class SplitWang(nn.Module):
    def __init__(self, variant: str, n_classes: int = 5, d: int = 64, gate_h: int = 32):
        super().__init__()
        assert variant in "CDE"
        w = ResNet1dWang(n_classes)
        mods = list(w.children())                                            # [conv, bn, relu, s1, s2, s3, head]
        self.shallow, self.deep = nn.Sequential(*mods[:4]), nn.Sequential(*mods[4:6])   # B0 head unused
        self.variant = variant
        self.p_local, self.p_ctx = nn.Linear(128, d), nn.Linear(128, d)
        if variant == "D":
            self.gate = _mlp(2 * d, gate_h)
        if variant == "E":
            self.P = nn.Linear(d, d)
            self.gate = _mlp(3 * d, gate_h)
        self.head = _head(2 * d if variant == "C" else d, n_classes)
        self.last_r = None

    def forward(self, x):
        h1 = self.shallow(x)
        fl, fc = self.p_local(h1.mean(-1)), self.p_ctx(self.deep(h1).mean(-1))
        if self.variant == "C":
            return self.head(torch.cat([fl, fc], 1))
        z = torch.cat([fl, fc], 1) if self.variant == "D" else torch.cat([fl, fc, (fl - self.P(fc)).abs()], 1)
        r = torch.sigmoid(self.gate(z))
        self.last_r = r.squeeze(1)
        return self.head(r * fl + (1 - r) * fc)


T2_MODELS = {"t2_C": lambda: SplitWang("C"), "t2_D": lambda: SplitWang("D"), "t2_E": lambda: SplitWang("E")}


@torch.no_grad()
def predict_with_r(model, X: torch.Tensor, device, crop: int, stride: int, bs: int = 256):
    """Sliding-window mean probabilities (as train.predict) plus per-record mean r (NaN if no gate)."""
    model.eval()
    starts = list(range(0, X.shape[-1] - crop + 1, stride))
    step = max(1, bs // len(starts))
    P, R = [], []
    for i in range(0, len(X), step):
        xb = X[i:i + step].to(device)
        w = torch.stack([xb[..., s:s + crop] for s in starts], 1).flatten(0, 1)
        p = torch.sigmoid(model(w)).view(len(xb), len(starts), -1).mean(1)
        r = getattr(model, "last_r", None)
        P.append(p.cpu())
        R.append(r.view(len(xb), len(starts)).mean(1).cpu() if r is not None else torch.full((len(xb),), float("nan")))
    return torch.cat(P).numpy(), torch.cat(R).numpy()
