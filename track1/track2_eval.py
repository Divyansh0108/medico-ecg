"""Evaluate models on the corruption grid (clean + family x SNR x mode) of fold 9.

Usage: python track2_eval.py --name difficulty --snrs 15 6 0 -6 --tags TAG [TAG ...]
       python track2_eval.py --name nstdb --nstdb-snrs 0 -6 --tags ...        (real noise, nstdb.py)
Each condition is generated once (fixed seeded RNG per record, corruptions.py) and every model predicts
on the same corrupted records. Writes results/track2/grid/{name}/{tag}.npz with
  conds (C,), probs (C, N, 5), r (C, N) (NaN when the model has no gate), y, patient_id, ecg_id.
Chapman models (RULES4.md S4, config dataset == "chapman") are scored on the Chapman test split; one call
takes models of one dataset only. --force-r R: the gate fusion weight is fixed to R (RULES4.md 5).
Fold 10 only with --fold10 --purpose "...", which logs every model to FOLD10_LOG.md first.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

import nstdb
from corruptions import cond_name, conditions, corrupt_split
from data import set_seed, standardize
from fold10_log import log_fold10
from train import HERE, build_model, classes_of, filtered
from track2_models import predict_with_r

RUN_DIRS = [os.path.join(HERE, "results", "track2", "runs"), os.path.join(HERE, "results", "crop")]


def run_json(tag: str) -> dict:
    for d in RUN_DIRS:
        p = os.path.join(d, f"{tag}.json")
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--snrs", nargs="*", type=float, default=[], help="synthetic benchmark SNRs")
    ap.add_argument("--nstdb-snrs", nargs="*", type=float, default=[], help="NSTDB benchmark SNRs (EVAL noise)")
    ap.add_argument("--force-r", type=float, default=None)
    ap.add_argument("--fold10", action="store_true")
    ap.add_argument("--purpose", default="")
    a = ap.parse_args()
    datasets = {getattr(SimpleNamespace(**run_json(t)["config"]), "dataset", "ptbxl") for t in a.tags}
    assert len(datasets) == 1, datasets
    dataset = datasets.pop()
    assert not (a.fold10 and dataset != "ptbxl")
    split = "test" if a.fold10 or dataset == "chapman" else "val"
    out_dir = os.path.join(HERE, "results", "track2", "grid", a.name + ("_fold10" if a.fold10 else ""))
    os.makedirs(out_dir, exist_ok=True)
    set_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = {}
    for tag in a.tags:
        c = SimpleNamespace(**run_json(tag)["config"])
        assert c.norm == "dataset" and not c.no_bandpass and c.agg == "mean", tag
        m = build_model(c.model, n_classes=len(classes_of(dataset))).to(device)
        m.load_state_dict(torch.load(os.path.join(HERE, "checkpoints", f"{tag}.pt"), map_location=device))
        if a.force_r is not None:
            assert hasattr(m, "force_r"), tag
            m.force_r = a.force_r
        models[tag] = (m, c.crop, c.stride)
    if a.fold10:
        if not a.purpose:
            raise SystemExit("--fold10 needs --purpose")
        for tag in a.tags:
            if os.path.exists(os.path.join(out_dir, f"{tag}.npz")):
                raise SystemExit(f"{tag} was already scored on fold 10 under '{a.name}'")
        for tag in a.tags:
            log_fold10(f"track2 grid '{a.name}': {tag}", [tag], a.purpose)
    d, mu, sd = filtered(dataset)
    Xf, Y, meta = d[split]
    ids = meta["ecg_id"].to_numpy()
    del d
    conds = [("clean", None, None)] + (conditions(a.snrs) if a.snrs else []) + \
        (nstdb.conditions(a.nstdb_snrs) if a.nstdb_snrs else [])
    names = ["clean"] + [cond_name(*c) for c in conds[1:]]
    P = {t: np.zeros((len(conds), len(Y), Y.shape[1]), np.float32) for t in models}
    R = {t: np.zeros((len(conds), len(Y)), np.float32) for t in models}
    t0 = time.time()
    for k, (fam, snr, mode) in enumerate(conds):
        Xc = (Xf if fam == "clean" else nstdb.corrupt_split(Xf, ids, fam, snr, mode) if fam.startswith("nstdb_")
              else corrupt_split(Xf, ids, fam, snr, mode))
        Xt = torch.from_numpy(standardize(Xc, mu, sd))
        for t, (m, crop, stride) in models.items():
            P[t][k], R[t][k] = predict_with_r(m, Xt, device, crop, stride)
        print(f"[{a.name}] {k + 1}/{len(conds)} {names[k]} {time.time() - t0:.0f}s", flush=True)
    for t in models:
        np.savez(os.path.join(out_dir, f"{t}.npz"), conds=np.array(names), probs=P[t], r=R[t], y=Y,
                 patient_id=meta["patient_id"].to_numpy(), ecg_id=ids)
    print(f"[{a.name}] wrote {len(models)} files to {out_dir}")


if __name__ == "__main__":
    main()
