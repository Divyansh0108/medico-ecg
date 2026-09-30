"""Post-hoc analysis on saved probabilities. Everything is fitted/tuned on fold 9 (val) only
and applied unchanged to fold 10 (test).

Usage: python calibrate.py [--model M1|M2]   (default: better seed-0 model by VAL macro-AUROC)
Writes results/calibration_s0.json.
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from data import SUPERCLASSES
from metrics import ece_binary, macro_auroc, patient_bootstrap_ci, per_class_auroc

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pick_model() -> str:
    v = {m: json.load(open(os.path.join(R, f"{m}_s0.json")))["val_macro_auroc"] for m in ("M1", "M2")}
    return max(v, key=v.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    m = a.model or pick_model()
    z = np.load(os.path.join(R, "probs", f"{m}_s0.npz"))
    pv, pt, yv, yt, pid = z["val"], z["test"], z["y_val"], z["y_test"], z["test_patient_id"]
    K = len(SUPERCLASSES)

    # 1-2. per-class Platt scaling (fit on val logits)
    pt_cal = np.zeros_like(pt)
    platt = {}
    for k in range(K):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv[:, k])[:, None], yv[:, k])
        pt_cal[:, k] = lr.predict_proba(logit(pt[:, k])[:, None])[:, 1]
        platt[SUPERCLASSES[k]] = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    ece_before = {c: ece_binary(yt[:, k], pt[:, k]) for k, c in enumerate(SUPERCLASSES)}
    ece_after = {c: ece_binary(yt[:, k], pt_cal[:, k]) for k, c in enumerate(SUPERCLASSES)}

    # 3. per-class F1-optimal thresholds on val (raw probs; Platt is monotone so equivalent)
    grid = np.linspace(0.01, 0.99, 99)
    thr = {}
    for k, c in enumerate(SUPERCLASSES):
        f1s = [f1_score(yv[:, k], pv[:, k] >= t, zero_division=0) for t in grid]
        thr[c] = float(grid[int(np.argmax(f1s))])
    t_vec = np.array([thr[c] for c in SUPERCLASSES])
    yhat = (pt >= t_vec).astype(int)
    sens = recall_score(yt, yhat, average=None, zero_division=0)
    spec = np.array([((yhat[:, k] == 0) & (yt[:, k] == 0)).sum() / (yt[:, k] == 0).sum() for k in range(K)])
    bal = (sens + spec) / 2

    # 4. patient-level bootstrap CI for test macro-AUROC
    lo, hi, _ = patient_bootstrap_ci(yt, pt, pid, n=1000, seed=0)

    res = {
        "model": m, "seed": 0, "selection": "better seed-0 model by val macro-AUROC" if a.model is None else "manual",
        "test_macro_auroc_raw": macro_auroc(yt, pt), "test_macro_auroc_calibrated": macro_auroc(yt, pt_cal),
        "test_per_class_auroc": per_class_auroc(yt, pt),
        "test_macro_auroc_ci95_patient_bootstrap": [lo, hi], "bootstrap_resamples": 1000,
        "ece_15bin": {"before": ece_before, "after": ece_after,
                      "mean_before": float(np.mean(list(ece_before.values()))),
                      "mean_after": float(np.mean(list(ece_after.values())))},
        "platt_params": platt,
        "thresholds_val_f1": thr,
        "test_threshold_metrics": {
            "macro_f1": float(f1_score(yt, yhat, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(yt, yhat, average="macro", zero_division=0)),
            "macro_recall": float(sens.mean()),
            "macro_balanced_accuracy": float(bal.mean()),
            "per_class": {c: {"f1": float(f1_score(yt[:, k], yhat[:, k], zero_division=0)),
                              "precision": float(precision_score(yt[:, k], yhat[:, k], zero_division=0)),
                              "recall": float(sens[k]), "specificity": float(spec[k]),
                              "balanced_accuracy": float(bal[k])} for k, c in enumerate(SUPERCLASSES)}},
        "note": "Platt and thresholds fitted on fold 9 only; fold 10 used for evaluation only.",
    }
    with open(os.path.join(R, "calibration_s0.json"), "w") as f:
        json.dump(res, f, indent=2)
    tm = res["test_threshold_metrics"]
    print(f"model {m}: test mAUROC {res['test_macro_auroc_raw']:.4f} (cal {res['test_macro_auroc_calibrated']:.4f}) "
          f"95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"ECE mean before {res['ece_15bin']['mean_before']:.4f} after {res['ece_15bin']['mean_after']:.4f}")
    print(f"macro-F1 {tm['macro_f1']:.4f} P {tm['macro_precision']:.4f} R {tm['macro_recall']:.4f} "
          f"BalAcc {tm['macro_balanced_accuracy']:.4f}  thresholds {thr}")


if __name__ == "__main__":
    main()
