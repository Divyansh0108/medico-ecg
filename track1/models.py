"""M1: 1D ResNet-SE. M2: M1 backbone + sinusoidal PE + 2-layer transformer encoder."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SE(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, c // r), nn.ReLU(inplace=True), nn.Linear(c // r, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1)).unsqueeze(-1)


class ResSEBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, k: int = 7):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, k, 1, k // 2, bias=False), nn.BatchNorm1d(cout), SE(cout))
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class Backbone(nn.Module):
    """Stem (stride 2) + 4 stages x 2 SE-residual blocks, first block of each stage strided.
    (12, 1000) -> (128, 32)."""
    def __init__(self, in_ch: int = 12, chans=(32, 64, 128, 128), blocks_per_stage: int = 2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, chans[0], 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(chans[0]), nn.ReLU(inplace=True))
        layers, cin = [], chans[0]
        for c in chans:
            for b in range(blocks_per_stage):
                layers.append(ResSEBlock(cin, c, stride=2 if b == 0 else 1))
                cin = c
        self.stages = nn.Sequential(*layers)
        self.out_ch = cin

    def forward(self, x):
        return self.stages(self.stem(x))


class M1(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.backbone = Backbone()
        self.head = nn.Linear(self.backbone.out_ch, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x).mean(-1))


def sinusoidal_pe(length: int, d: int) -> torch.Tensor:
    pos = torch.arange(length).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class M2(nn.Module):
    def __init__(self, n_classes: int = 5, d_model: int = 128, nhead: int = 4, ff: int = 256,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = Backbone()
        assert self.backbone.out_ch == d_model
        self.register_buffer("pe", sinusoidal_pe(512, d_model), persistent=False)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.backbone(x).transpose(1, 2)          # (B, T, d)
        h = self.encoder(h + self.pe[: h.size(1)])
        return self.head(h.mean(1))


MODELS = {"M1": M1, "M2": M2}
