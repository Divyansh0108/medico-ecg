"""Seeds 0-4 summary + probability-averaged seed ensemble.  Usage: python ensemble.py M1 [M2]"""
import json
import os
import sys

import numpy as np

from metrics import macro_auroc, patient_bootstrap_ci, per_class_auroc

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
out = {}
for m in sys.argv[1:]:
    seeds = [s for s in range(5) if os.path.exists(os.path.join(R, "probs", f"{m}_s{s}.npz"))]
    zs = [np.load(os.path.join(R, "probs", f"{m}_s{s}.npz")) for s in seeds]
    aucs = np.array([json.load(open(os.path.join(R, f"{m}_s{s}.json")))["test_macro_auroc"] for s in seeds])
    yt, pid = zs[0]["y_test"], zs[0]["test_patient_id"]
    assert all(np.array_equal(z["test_ecg_id"], zs[0]["test_ecg_id"]) for z in zs)
    pe = np.mean([z["test"] for z in zs], 0)
    lo, hi, _ = patient_bootstrap_ci(yt, pe, pid, n=1000, seed=0)
    out[m] = {"seeds": seeds, "test_macro_auroc_per_seed": aucs.tolist(),
              "mean": float(aucs.mean()), "std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
              "ensemble_test_macro_auroc": macro_auroc(yt, pe), "ensemble_ci95_patient_bootstrap": [lo, hi],
              "ensemble_per_class_auroc": per_class_auroc(yt, pe)}
    o = out[m]
    print(f"{m}: seeds {seeds} test mAUROC {o['mean']:.4f} +- {o['std']:.4f} (per seed {np.round(aucs, 4).tolist()})")
    print(f"   single-model mean vs xresnet1d101 .928: {o['mean'] - .928:+.4f}")
    print(f"   {len(seeds)}-seed ensemble {o['ensemble_test_macro_auroc']:.4f} 95% CI [{lo:.4f}, {hi:.4f}]"
          f"  vs published ensemble .934: {o['ensemble_test_macro_auroc'] - .934:+.4f}")
with open(os.path.join(R, "seeds_ensemble.json"), "w") as f:
    json.dump(out, f, indent=2)
