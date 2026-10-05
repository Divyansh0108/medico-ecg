"""CinC 2017 transfer (RULES2.md section 3B-D): data, windows and single-lead inference (no training, no tuning).
Writes results/robustness/cinc2017/{data.npz, <norm>/<tag>.npz} with per-recording r_mean, r_min, feat (mean over
windows of the classifier input) and prob (mean over windows)."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import scipy.io as sio
import wfdb
from scipy.signal import resample_poly

from loaders.ptbxl import bandpass
from external.single_lead_infer import SL, device, h1, h2, load_model, score_windows, windows
from training.train import HERE

ROOT = os.path.join(HERE, "..", "data", "cinc2017")
OUT = os.path.join(HERE, "results", "robustness", "cinc2017")


def load_records():
    cp = os.path.join(OUT, "data.npz")
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        return list(z["ids"]), [x for x in z["signals"]], z
    ref = pd.read_csv(os.path.join(ROOT, "training2017", "REFERENCE.csv"), header=None, names=["id", "label"])
    v3 = pd.read_csv(os.path.join(ROOT, "REFERENCE-v3.csv"), header=None, names=["id", "label"]).set_index("id")["label"]
    sig, fs_seen = [], set()
    for rid in ref["id"]:
        hdr = wfdb.rdheader(os.path.join(ROOT, "training2017", rid))
        fs_seen.add(hdr.fs)
        x = sio.loadmat(os.path.join(ROOT, "training2017", f"{rid}.mat"))["val"][0].astype(np.float64) / 1000.0  # mV
        assert len(x) == hdr.sig_len
        sig.append(bandpass(resample_poly(x, 1, 3)))
    assert fs_seen == {300}, fs_seen
    os.makedirs(OUT, exist_ok=True)
    np.savez(cp, ids=ref["id"].to_numpy(), label=ref["label"].to_numpy(), label_v3=v3.loc[ref["id"]].to_numpy(),
             signals=np.array(sig, dtype=object), allow_pickle=True)
    z = np.load(cp, allow_pickle=True)
    return list(z["ids"]), [x for x in z["signals"]], z


def main():
    ids, sig, z = load_records()
    lens = np.array([len(x) for x in sig])
    print(f"CinC2017: {len(ids)} recordings, fs 300 Hz -> 100 Hz; length {lens.min() / 100:.1f}-{lens.max() / 100:.1f} s "
          f"(median {np.median(lens) / 100:.1f}); labels {dict(zip(*np.unique(z['label'], return_counts=True)))}")
    allx = np.concatenate(sig)
    mu, sd = float(allx.mean()), float(allx.std())
    print(f"pooled mean {mu:.5f} sd {sd:.5f} (all samples, labels not used)")
    W = [windows(x) for x in sig]
    nwin = np.array([len(w) for w in W])
    off = np.concatenate([[0], np.cumsum(nwin)[:-1]])
    print(f"windows: {nwin.sum()} (per record {nwin.min()}-{nwin.max()}), {np.sum(lens < 1000)} records < 10 s reflect-padded")
    Xw = np.concatenate(W).astype(np.float64)
    norms = {"pooled": ((Xw - mu) / sd).astype(np.float32)[:, None],
             "record": np.concatenate([((w - x.mean()) / (x.std() + 1e-12)) for w, x in zip(W, sig)]).astype(np.float32)[:, None]}
    np.savez(os.path.join(OUT, "heuristics.npz"), H1=np.array([h1(x) for x in sig]), H2=np.array([h2(x) for x in sig]),
             nwin=nwin, length=lens, mu=mu, sd=sd)
    dev = device()
    for norm, X in norms.items():
        os.makedirs(os.path.join(OUT, norm), exist_ok=True)
        for tags in SL.values():
            for t in tags:
                fp = os.path.join(OUT, norm, f"{t}.npz")
                if os.path.exists(fp):
                    continue
                t0 = time.time()
                m, crop, stride = load_model(t, dev)
                r, p, f = score_windows(m, X, dev, crop, stride)
                agg = lambda a: np.add.reduceat(a, off, axis=0) / nwin.reshape((-1,) + (1,) * (a.ndim - 1))
                r_min = np.minimum.reduceat(r, off)
                np.savez(fp, r_mean=agg(r), r_min=r_min, prob=agg(p), feat=agg(f) if norm == "pooled" else np.zeros(0))
                print(f"[{norm}] {t} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
