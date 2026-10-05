"""Backbones from the PTB-XL benchmark (Strodthoff et al. 2021, github.com/helme/ecg_ptbxl_benchmarking,
code/models/{resnet1d,xresnet1d,inception1d,basic_conv1d}.py), re-implemented in plain PyTorch (no fastai).
Constructor defaults follow fastai_model.py: kernel_size=5, lin_ftrs_head=[128], ps_head=0.5,
concat pooling, inception kernel_size=8*5=40.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------- head (basic_conv1d.create_head1d)
class AdaptiveConcatPool1d(nn.Module):
    def forward(self, x):
        return torch.cat([x.amax(-1), x.mean(-1)], 1)


def create_head1d(nf: int, nc: int, lin_ftrs=(128,), ps: float = 0.5) -> nn.Sequential:
    """ConcatPool -> [BN, Dropout(ps/2), Linear, ReLU] -> [BN, Dropout(ps), Linear] (fastai bn_drop_lin)."""
    ftrs = [2 * nf, *lin_ftrs, nc]
    drops = [ps / 2] * (len(ftrs) - 2) + [ps]
    layers = [AdaptiveConcatPool1d()]
    for i, (ni, no, p) in enumerate(zip(ftrs[:-1], ftrs[1:], drops)):
        layers += [nn.BatchNorm1d(ni), nn.Dropout(p), nn.Linear(ni, no)]
        if i < len(ftrs) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------- resnet1d_wang (resnet1d.py)
def _conv(cin, cout, stride=1, k=3):
    return nn.Conv1d(cin, cout, k, stride, (k - 1) // 2, bias=False)


class BasicBlock1d(nn.Module):
    def __init__(self, cin, cout, stride=1, k=(3, 3), downsample=None):
        super().__init__()
        if isinstance(k, int):
            k = (k, k // 2 + 1)
        self.conv1, self.bn1 = _conv(cin, cout, stride, k[0]), nn.BatchNorm1d(cout)
        self.conv2, self.bn2 = _conv(cout, cout, 1, k[1]), nn.BatchNorm1d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        r = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + r)


class ResNet1dWang(nn.Sequential):
    """resnet1d_wang: stem k7 s1 no pool, 3 stages x 1 BasicBlock, 128 ch fixed, kernels [5,3]."""
    def __init__(self, n_classes=5, in_ch=12, planes=128, layers=(1, 1, 1), k=(5, 3), k_stem=7):
        mods = [nn.Conv1d(in_ch, planes, k_stem, 1, (k_stem - 1) // 2, bias=False),
                nn.BatchNorm1d(planes), nn.ReLU(inplace=True)]
        for i, n in enumerate(layers):
            stride = 1 if i == 0 else 2
            ds = None if stride == 1 else nn.Sequential(nn.Conv1d(planes, planes, 1, stride, bias=False),
                                                        nn.BatchNorm1d(planes))
            mods.append(nn.Sequential(BasicBlock1d(planes, planes, stride, list(k), ds),
                                      *[BasicBlock1d(planes, planes) for _ in range(1, n)]))
        mods.append(create_head1d(planes, n_classes))
        super().__init__(*mods)


# ---------------------------------------------------------------- xresnet1d (xresnet1d.py)
def _bn(nf, zero=False):
    bn = nn.BatchNorm1d(nf)
    bn.bias.data.fill_(1e-3)
    bn.weight.data.fill_(0.0 if zero else 1.0)
    return bn


def conv_layer(ni, nf, ks=3, stride=1, act=True, zero_bn=False):
    """fastai ConvLayer, bn_1st=True: conv -> BN -> ReLU."""
    layers = [nn.Conv1d(ni, nf, ks, stride, (ks - 1) // 2, bias=False), _bn(nf, zero_bn)]
    if act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class XResBlock(nn.Module):
    def __init__(self, expansion, ni, nf, stride=1, ks=5):
        super().__init__()
        nh = nf
        nf, ni = nf * expansion, ni * expansion
        if expansion == 1:
            convs = [conv_layer(ni, nh, ks, stride), conv_layer(nh, nf, ks, act=False, zero_bn=True)]
        else:
            convs = [conv_layer(ni, nh, 1), conv_layer(nh, nh, ks, stride),
                     conv_layer(nh, nf, 1, act=False, zero_bn=True)]
        self.convpath = nn.Sequential(*convs)
        idpath = []
        if stride != 1:
            idpath.append(nn.AvgPool1d(2, ceil_mode=True))
        if ni != nf:
            idpath.append(conv_layer(ni, nf, 1, act=False))
        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.convpath(x) + self.idpath(x))


def _init_cnn(m):
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for c in m.children():
        _init_cnn(c)


class XResNet1d(nn.Sequential):
    def __init__(self, expansion, layers, n_classes=5, in_ch=12, stem=(32, 32, 64), ks=5, ks_stem=5):
        szs = [in_ch, *stem]
        mods = [conv_layer(szs[i], szs[i + 1], ks_stem, 2 if i == 0 else 1) for i in range(3)]
        mods.append(nn.MaxPool1d(3, 2, 1))
        block_szs = [64 // expansion, 64, 64, 64, 64] + [32] * (len(layers) - 4)
        for i, n in enumerate(layers):
            mods.append(nn.Sequential(*[
                XResBlock(expansion, block_szs[i] if j == 0 else block_szs[i + 1], block_szs[i + 1],
                          stride=(1 if i == 0 else 2) if j == 0 else 1, ks=ks) for j in range(n)]))
        mods.append(create_head1d(block_szs[len(layers)] * expansion, n_classes))
        super().__init__(*mods)
        _init_cnn(self)


def xresnet1d50(n_classes=5):
    return XResNet1d(4, [3, 4, 6, 3], n_classes)


def xresnet1d101(n_classes=5):
    return XResNet1d(4, [3, 4, 23, 3], n_classes)


# ---------------------------------------------------------------- inception1d (inception1d.py)
class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, bottleneck=32):
        super().__init__()
        self.bottleneck = _conv(ni, bottleneck, 1, 1)
        self.convs = nn.ModuleList([_conv(bottleneck, nb_filters, 1, k) for k in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, 1, padding=1), _conv(ni, nb_filters, 1, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss) + 1) * nb_filters), nn.ReLU())

    def forward(self, x):
        b = self.bottleneck(x)
        return self.bn_relu(torch.cat([c(b) for c in self.convs] + [self.conv_bottle(x)], 1))


class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.conv, self.bn, self.act = _conv(ni, nf, 1, 1), nn.BatchNorm1d(nf), nn.ReLU(True)

    def forward(self, inp, out):
        return self.act(out + self.bn(self.conv(inp)))


class Inception1d(nn.Module):
    def __init__(self, n_classes=5, in_ch=12, kernel_size=40, depth=6, bottleneck=32, nb_filters=32):
        super().__init__()
        kss = [k - 1 if k % 2 == 0 else k for k in (kernel_size, kernel_size // 2, kernel_size // 4)]  # 39,19,9
        nf = (len(kss) + 1) * nb_filters
        self.depth = depth
        self.im = nn.ModuleList([InceptionBlock1d(in_ch if d == 0 else nf, nb_filters, kss, bottleneck)
                                 for d in range(depth)])
        self.sk = nn.ModuleList([Shortcut1d(in_ch if d == 0 else nf, nf) for d in range(depth // 3)])
        self.head = create_head1d(nf, n_classes)

    def forward(self, x):
        res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if d % 3 == 2:
                x = self.sk[d // 3](res, x)
                res = x
        return self.head(x)


STRODTHOFF = {"resnet1d_wang": ResNet1dWang, "xresnet1d50": xresnet1d50,
              "xresnet1d101": xresnet1d101, "inception1d": Inception1d}
