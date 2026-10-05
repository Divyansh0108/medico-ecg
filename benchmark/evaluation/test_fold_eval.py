"""Evaluate fold 10 ONCE for runs chosen on fold 9. Several tags = probability-averaged ensemble.

Usage: python -m evaluation.test_fold_eval --name FINAL --tags xresnet1d101_crop [more tags...]
Rebuilds each run's model + preprocessing from {dir}/{tag}.json, loads checkpoints/{tag}.pt, predicts
fold 10 with the run's own eval settings, writes {dir}/{name}.json. Refuses to overwrite an existing result.
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from loaders.ptbxl import SUPERCLASSES, set_seed
from evaluation.metrics import macro_auroc, patient_bootstrap_ci, per_class_auroc
from training.train import HERE, build_model, get_data, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--dir", default=os.path.join(HERE, "results", "ptbxl_baselines"))
    a = ap.parse_args()
    out = os.path.join(a.dir, f"{a.name}.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: fold 10 was already evaluated for '{a.name}'")
    set_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pv, pt, runs = [], [], []
    for tag in a.tags:
        r = json.load(open(os.path.join(a.dir, f"{tag}.json")))
        c = SimpleNamespace(**r["config"])
        _, (Xva, Yva, _), (Xte, Yte, mte) = get_data(c)
        model = build_model(c.model).to(device)
        model.load_state_dict(torch.load(os.path.join(HERE, "checkpoints", f"{tag}.pt"), map_location=device))
        ev = dict(crop=c.crop, stride=c.stride, agg=c.agg)
        pv.append(predict(model, torch.from_numpy(Xva), device, **ev))
        pt.append(predict(model, torch.from_numpy(Xte), device, **ev))
        runs.append({"tag": tag, "val": macro_auroc(Yva, pv[-1]), "test": macro_auroc(Yte, pt[-1])})
        print(f"{tag}: val {runs[-1]['val']:.4f} test {runs[-1]['test']:.4f}", flush=True)
    pva, pte = np.mean(pv, 0), np.mean(pt, 0)
    lo, hi, _ = patient_bootstrap_ci(Yte, pte, mte["patient_id"].to_numpy())
    res = {"name": a.name, "tags": a.tags, "runs": runs, "classes": SUPERCLASSES,
           "val_macro_auroc": macro_auroc(Yva, pva), "test_macro_auroc": macro_auroc(Yte, pte),
           "test_ci95_patient_bootstrap": [lo, hi], "test_per_class_auroc": per_class_auroc(Yte, pte)}
    json.dump(res, open(out, "w"), indent=2)
    np.savez(os.path.join(a.dir, "probs", f"{a.name}_test.npz"), val=pva, test=pte, y_test=Yte,
             test_ecg_id=mte["ecg_id"].to_numpy(), test_patient_id=mte["patient_id"].to_numpy())
    print(f"[{a.name}] val {res['val_macro_auroc']:.4f} TEST {res['test_macro_auroc']:.4f} "
          f"CI [{lo:.4f}, {hi:.4f}]", flush=True)


if __name__ == "__main__":
    main()
