"""Single-lead inference on 10 s windows for the external datasets (RULES2.md sections 3 and 5).

Each 10 s window (1, 1000) is scored like a PTB-XL record: 2.5 s crops with stride 1.25 s, mean over crops.
Returns per window: r (gated models; NaN otherwise), the 5 class probabilities, and the feature vector that
enters the classifier (B0: 256-d concat pool of the head; D/E/F: the 64-d fused feature), averaged over crops.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from scipy.signal import welch

from loaders.ptbxl import FS
from training.train import HERE, build_model

SL = {k: [f"SL_{p}_s{s}" for s in (0, 1, 2)] for k, p in
      [("B0-clean", "B0clean"), ("B0-aug", "B0aug"), ("D", "D"), ("E", "E"), ("F", "F"), ("E-clean", "Eclean")]}
GATED = ["D", "E", "F", "E-clean"]
WIN, STRIDE = 1000, 500


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(tag: str, dev):
    c = SimpleNamespace(**json.load(open(os.path.join(HERE, "results", "robustness", "runs_sl", f"{tag}.json")))["config"])
    assert c.leads == [0] and c.norm == "dataset" and c.crop == 250 and c.agg == "mean", tag
    m = build_model(c.model, c.leads).to(dev)
    m.load_state_dict(torch.load(os.path.join(HERE, "checkpoints", f"{tag}.pt"), map_location=dev))
    m.eval()
    return m, c.crop, c.stride


@torch.no_grad()
def score_windows(model, X: np.ndarray, dev, crop: int = 250, stride: int = 125, bs: int = 512):
    """X: (N, 1, 1000) standardized float32 -> r (N,), probs (N, 5), feats (N, d)."""
    feats = {}
    head = model.head if hasattr(model, "variant") else model[-1]
    if hasattr(model, "variant"):       # SplitWang: the fused feature is the head's input
        h = head.register_forward_pre_hook(lambda mod, inp: feats.__setitem__("f", inp[0]))
    else:                               # ResNet1dWang: output of AdaptiveConcatPool1d (head[0])
        h = head[0].register_forward_hook(lambda mod, inp, out: feats.__setitem__("f", out))
    starts = list(range(0, X.shape[-1] - crop + 1, stride))
    step = max(1, bs // len(starts))
    R, P, F = [], [], []
    try:
        for i in range(0, len(X), step):
            xb = torch.from_numpy(X[i:i + step]).to(dev)
            w = torch.stack([xb[..., s:s + crop] for s in starts], 1).flatten(0, 1)
            p = torch.sigmoid(model(w)).view(len(xb), len(starts), -1).mean(1)
            f = feats["f"].view(len(xb), len(starts), -1).mean(1)
            r = getattr(model, "last_r", None)
            R.append(r.view(len(xb), len(starts)).mean(1).cpu() if r is not None else torch.full((len(xb),), float("nan")))
            P.append(p.cpu())
            F.append(f.cpu())
    finally:
        h.remove()
    return torch.cat(R).numpy(), torch.cat(P).numpy(), torch.cat(F).numpy()


def windows(x: np.ndarray) -> np.ndarray:
    """(T,) -> (W, 1000): starts 0, 500, ... while the window fits, plus a final window ending at the last sample;
    records shorter than 10 s are reflect-padded (repeated if needed) to 1000 samples."""
    T = len(x)
    if T < WIN:
        assert T >= 2, T
        while len(x) < WIN:
            x = np.pad(x, (0, min(WIN - len(x), len(x) - 1)), mode="reflect")
        return x[None, :WIN]
    starts = list(range(0, T - WIN + 1, STRIDE))
    if starts[-1] + WIN < T:
        starts.append(T - WIN)
    return np.stack([x[s:s + WIN] for s in starts])


def h1(x: np.ndarray) -> float:
    """power(20-40 Hz) / power(0.5-20 Hz), Welch PSD (nperseg 256) of the 100 Hz band-passed signal."""
    f, p = welch(x, fs=FS, nperseg=min(256, len(x)))
    return float(p[(f >= 20) & (f <= 40)].sum() / p[(f >= 0.5) & (f < 20)].sum())


def h2(x: np.ndarray) -> float:
    """max |z-scored amplitude| (per-record z-score)."""
    return float(np.max(np.abs((x - x.mean()) / (x.std() + 1e-12))))
