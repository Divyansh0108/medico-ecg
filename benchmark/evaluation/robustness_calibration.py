"""Track 2, step 2: calibration and thresholds for resnet1d_wang_crop_dsnorm seeds 0-2 and their 3-seed
probability average. No retraining. Platt scaling and thresholds are fitted on fold 9 only and applied
unchanged to fold 10. Each fold-10 scoring is logged (evaluation/fold10_log.py).
Writes results/robustness/calibration.json and results/robustness/calibration.md (table for REPORT.md)."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, precision_score

from loaders.ptbxl import SUPERCLASSES, set_seed
from evaluation.fold10_log import log_fold10
from evaluation.metrics import ece_binary, macro_auroc, patient_bootstrap_ci
from training.train import HERE, build_model, get_data, predict

T1 = os.path.join(HERE, "results", "ptbxl_baselines")
OUT = os.path.join(HERE, "results", "robustness")
TAGS = ["resnet1d_wang_crop_dsnorm", "resnet1d_wang_crop_dsnorm_s1", "resnet1d_wang_crop_dsnorm_s2"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def analyse(pv, yv, pt, yt, pid):
    K = len(SUPERCLASSES)
    pv_cal, pt_cal, platt, thr = np.zeros_like(pv), np.zeros_like(pt), {}, {}
    for k, c in enumerate(SUPERCLASSES):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv[:, k])[:, None], yv[:, k])
        pv_cal[:, k] = lr.predict_proba(logit(pv[:, k])[:, None])[:, 1]
        pt_cal[:, k] = lr.predict_proba(logit(pt[:, k])[:, None])[:, 1]
        platt[c] = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        pr, rc, th = precision_recall_curve(yv[:, k], pv_cal[:, k])       # max-F1 threshold on fold 9
        f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
        thr[c] = float(th[int(np.argmax(f1))])
    yhat = (pt_cal >= np.array([thr[c] for c in SUPERCLASSES])).astype(int)
    sens = np.array([(yhat[:, k] & (yt[:, k] == 1)).sum() / (yt[:, k] == 1).sum() for k in range(K)])
    spec = np.array([((yhat[:, k] == 0) & (yt[:, k] == 0)).sum() / (yt[:, k] == 0).sum() for k in range(K)])
    lo, hi, _ = patient_bootstrap_ci(yt, pt, pid, n=1000, seed=0)
    ece = lambda y, p: {c: ece_binary(y[:, k], p[:, k]) for k, c in enumerate(SUPERCLASSES)}
    eb, ea = ece(yt, pt), ece(yt, pt_cal)
    return {
        "fold10_macro_auroc_raw": macro_auroc(yt, pt), "fold10_macro_auroc_calibrated": macro_auroc(yt, pt_cal),
        "fold10_macro_auroc_ci95_patient_bootstrap": [lo, hi],
        "fold9_macro_auroc": macro_auroc(yv, pv),
        "ece_15bin_fold10": {"before": eb, "after": ea, "mean_before": float(np.mean(list(eb.values()))),
                             "mean_after": float(np.mean(list(ea.values())))},
        "ece_15bin_fold9_mean": {"before": float(np.mean(list(ece(yv, pv).values()))),
                                 "after": float(np.mean(list(ece(yv, pv_cal).values())))},
        "platt": platt, "thresholds_fold9_maxF1_on_calibrated": thr,
        "fold10_at_threshold": {
            "macro_f1": float(f1_score(yt, yhat, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(yt, yhat, average="macro", zero_division=0)),
            "macro_recall": float(sens.mean()), "macro_balanced_accuracy": float(((sens + spec) / 2).mean()),
            "per_class": {c: {"f1": float(f1_score(yt[:, k], yhat[:, k], zero_division=0)),
                              "precision": float(precision_score(yt[:, k], yhat[:, k], zero_division=0)),
                              "recall": float(sens[k]), "specificity": float(spec[k]),
                              "balanced_accuracy": float((sens[k] + spec[k]) / 2)} for k, c in enumerate(SUPERCLASSES)}},
    }


def main():
    set_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    c0 = SimpleNamespace(**json.load(open(os.path.join(T1, f"{TAGS[0]}.json")))["config"])
    _, (Xva, Yva, _), (Xte, Yte, mte) = get_data(c0)
    pid = mte["patient_id"].to_numpy()
    pv, pt = {}, {}
    for tag in TAGS:
        c = SimpleNamespace(**json.load(open(os.path.join(T1, f"{tag}.json")))["config"])
        assert (c.norm, c.no_bandpass, c.crop, c.stride, c.agg) == (c0.norm, c0.no_bandpass, c0.crop, c0.stride, c0.agg)
        m = build_model(c.model).to(device)
        m.load_state_dict(torch.load(os.path.join(HERE, "checkpoints", f"{tag}.pt"), map_location=device))
        ev = dict(crop=c.crop, stride=c.stride, agg=c.agg)
        pv[tag] = predict(m, torch.from_numpy(Xva), device, **ev)
        saved = np.load(os.path.join(T1, "probs", f"{tag}.npz"))["val"]
        assert np.abs(pv[tag] - saved).max() < 1e-4, "fold-9 predictions differ from the saved run"
        log_fold10(f"calibration: {tag}", [tag], "Track 2 step 2 calibration (no tuning on fold 10)")
        pt[tag] = predict(m, torch.from_numpy(Xte), device, **ev)
    log_fold10("calibration: 3-seed average", TAGS, "Track 2 step 2 calibration, mean of the 3 probabilities above")
    pv["ensemble_s0-2"] = np.mean([pv[t] for t in TAGS], 0)
    pt["ensemble_s0-2"] = np.mean([pt[t] for t in TAGS], 0)
    res = {"note": "Platt (per class, on logit p) and max-F1 thresholds fitted on fold 9 only; fold 10 only scored.",
           "classes": SUPERCLASSES, "entries": {k: analyse(pv[k], Yva, pt[k], Yte, pid) for k in pv}}
    json.dump(res, open(os.path.join(OUT, "calibration.json"), "w"), indent=2)

    L = ["| entry | fold-10 mAUROC (raw = calibrated) | 95% CI | ECE before -> after (mean of 5) | macro-F1 | precision | recall | bal. acc. |",
         "|---|---|---|---|---|---|---|---|"]
    for k, r in res["entries"].items():
        t, e, (lo, hi) = r["fold10_at_threshold"], r["ece_15bin_fold10"], r["fold10_macro_auroc_ci95_patient_bootstrap"]
        assert abs(r["fold10_macro_auroc_raw"] - r["fold10_macro_auroc_calibrated"]) < 1e-9
        L.append(f"| {k} | {r['fold10_macro_auroc_raw']:.4f} | {lo:.4f}-{hi:.4f} | {e['mean_before']:.4f} -> {e['mean_after']:.4f} | "
                 f"{t['macro_f1']:.4f} | {t['macro_precision']:.4f} | {t['macro_recall']:.4f} | {t['macro_balanced_accuracy']:.4f} |")
    L += ["", "Per-class ECE on fold 10 (15 bins), before -> after Platt:", "",
          "| entry | " + " | ".join(SUPERCLASSES) + " |", "|---" * (len(SUPERCLASSES) + 1) + "|"]
    for k, r in res["entries"].items():
        e = r["ece_15bin_fold10"]
        L.append(f"| {k} | " + " | ".join(f"{e['before'][c]:.3f} -> {e['after'][c]:.3f}" for c in SUPERCLASSES) + " |")
    L += ["", "Thresholds (max F1 on fold 9, calibrated probabilities), applied unchanged to fold 10:", "",
          "| entry | " + " | ".join(SUPERCLASSES) + " |", "|---" * (len(SUPERCLASSES) + 1) + "|"]
    for k, r in res["entries"].items():
        L.append(f"| {k} | " + " | ".join(f"{r['thresholds_fold9_maxF1_on_calibrated'][c]:.3f}" for c in SUPERCLASSES) + " |")
    open(os.path.join(OUT, "calibration.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
