"""BUT QDB (RULES2.md section 5): windows, motion, heuristics, single-lead r. No training, no tuning.
Stage 1 (prep): python butqdb.py prep  -> results/track2/butqdb/windows.npz
Stage 2 (infer): python butqdb.py infer -> results/track2/butqdb/<norm>/<tag>.npz (per-window r)"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

from data import bandpass
from sl_infer import GATED, SL, device, h1, h2, load_model, score_windows
from train import HERE

ROOT = os.path.join(HERE, "..", "data", "butqdb")
OUT = os.path.join(HERE, "results", "track2", "butqdb")
W1000 = 10_000          # 10 s at 1000 Hz


def prep():
    recs = [r.split("/")[0] for r in open(os.path.join(ROOT, "RECORDS")).read().split() if r.endswith("_ECG")]
    hp = butter(4, 0.5, btype="highpass", fs=100, output="sos")
    X, cls, rec, acc_rms, H1, H2, log = [], [], [], [], [], [], []
    for r in recs:
        he = wfdb.rdheader(os.path.join(ROOT, r, f"{r}_ECG"))
        ha = wfdb.rdheader(os.path.join(ROOT, r, f"{r}_ACC"))
        assert he.fs == 1000 and he.n_sig == 1 and ha.fs == 100 and ha.n_sig == 3, r
        a = pd.read_csv(os.path.join(ROOT, r, f"{r}_ANN.csv"), header=None).iloc[:, 9:12].dropna().astype(np.int64)
        a.columns = ["s", "e", "q"]
        assert a.e.max() == he.sig_len and (a.s.values[1:] == a.e.values[:-1] + 1).all(), r
        lab = np.zeros(he.sig_len, np.int8)                                  # sample-by-sample consensus (A1, A2)
        for s, e, q in a.itertuples(index=False):
            lab[s - 1:e] = q
        nw = he.sig_len // W1000
        L = lab[:nw * W1000].reshape(nw, W1000)
        lo, hi = L.min(1), L.max(1)
        touch = hi > 0
        keep = touch & (lo == hi)                                            # one class in {1,2,3} (A3)
        ecg = wfdb.rdrecord(os.path.join(ROOT, r, f"{r}_ECG")).p_signal[:, 0]
        x = bandpass(resample_poly(ecg, 1, 10))                              # A4
        del ecg
        accr = wfdb.rdrecord(os.path.join(ROOT, r, f"{r}_ACC")).p_signal
        mag = sosfiltfilt(hp, np.sqrt((accr ** 2).sum(1)))                   # A6
        idx = np.flatnonzero(keep)
        for i in idx:
            w = x[i * 1000:(i + 1) * 1000]
            X.append(w.astype(np.float32))
            m = mag[i * 1000:(i + 1) * 1000]
            acc_rms.append(float(np.sqrt(np.mean(m * m))))
            H1.append(h1(w))
            H2.append(h2(w))
        cls += list(lo[idx])
        rec += [r] * len(idx)
        c = {q: int((lo[idx] == q).sum()) for q in (1, 2, 3)}
        log.append(dict(record=r, subject=r[:3], ecg_fs=he.fs, ecg_len=he.sig_len, acc_fs=ha.fs, acc_len=ha.sig_len,
                        grid_touching=int(touch.sum()), kept=len(idx), dropped=int(touch.sum() - len(idx)),
                        dropped_mixed=int((touch & (lo != hi) & (lo > 0)).sum()),
                        dropped_partly_unannotated=int((touch & (lo == 0)).sum()), **{f"class{q}": v for q, v in c.items()}))
        print(log[-1], flush=True)
        del x, accr, mag
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "windows.npz"), X=np.stack(X), cls=np.array(cls), rec=np.array(rec),
             subject=np.array([r[:3] for r in rec]), acc_rms=np.array(acc_rms), H1=np.array(H1), H2=np.array(H2))
    pd.DataFrame(log).to_csv(os.path.join(OUT, "windows_log.csv"), index=False)


def infer():
    z = np.load(os.path.join(OUT, "windows.npz"))
    X = z["X"].astype(np.float64)
    mu, sd = float(X.mean()), float(X.std())
    norms = {"pooled": ((X - mu) / sd).astype(np.float32)[:, None],
             "window": ((X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)).astype(np.float32)[:, None]}
    np.savez(os.path.join(OUT, "norm.npz"), mu=mu, sd=sd)
    dev = device()
    for norm, Xn in norms.items():
        os.makedirs(os.path.join(OUT, norm), exist_ok=True)
        for k in GATED:
            for t in SL[k]:
                fp = os.path.join(OUT, norm, f"{t}.npz")
                if os.path.exists(fp):
                    continue
                t0 = time.time()
                m, crop, stride = load_model(t, dev)
                r, _, _ = score_windows(m, Xn, dev, crop, stride)
                np.savez(fp, r=r)
                print(f"[{norm}] {t} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"prep": prep, "infer": infer}[sys.argv[1]]()
