"""Round 3 report. Fold 9: seeds 0-4 (mean +- std, seed ensemble), seed-0 variants, the pre-specified
15-model ensemble. Fold 10 (only from evaluation/test_fold_eval.py outputs): final entries, CIs and the paired patient
bootstrap against the reproduced baselines. Writes results/ptbxl_baselines/REPORT_seeds.md.
Usage: python -m reporting.seed_report [--final]   (--final adds the fold-10 section; run it after evaluation/test_fold_eval.py)"""
import json
import os
import sys

import numpy as np

from evaluation.metrics import macro_auroc, paired_patient_bootstrap

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "ptbxl_baselines")
BACKBONES = ["resnet1d_wang", "inception1d", "xresnet1d50"]
VARIANTS = [("base (best epoch, early stopping)", "resnet1d_wang_crop_dsnorm"),
            ("SWA, epochs 41-50", "resnet1d_wang_crop_dsnorm_swa"),
            ("EMA 0.999, epochs 41-50", "resnet1d_wang_crop_dsnorm_ema"),
            ("label smoothing 0.05", "resnet1d_wang_crop_dsnorm_ls05"),
            ("crop 500, stride 250", "resnet1d_wang_crop500_dsnorm")]
FINALS = ["FINAL2_single", "FINAL2_ensemble"]
BASELINES = [("resnet1d_wang (reproduced)", "FINAL_single", .930), ("xresnet1d101 (reproduced)", "BASE_xresnet1d101", .928)]


def tags(m):
    return [f"{m}_crop_dsnorm"] + [f"{m}_crop_dsnorm_s{s}" for s in range(1, 5)]


def js(t):
    return json.load(open(os.path.join(D, f"{t}.json")))


def val(t):
    z = np.load(os.path.join(D, "probs", f"{t}.npz"))
    return z["y_val"], z["val"]


L = ["# PTB-XL Track 1 - seeds, SWA/EMA, label smoothing, crop 500, ensemble", "",
     "Recipe: 2.5 s crops, sliding-window mean (250/125), dataset norm fitted on train, band-pass on.",
     "Selection on fold 9 only; rule and decision in SELECTION2.md.", "",
     "## 1. Seeds 0-4 (fold 9)", "",
     "| model | per seed (0-4) | mean +- std | 5-seed probability average |", "|---|---|---|---|"]
allp = []
for m in BACKBONES:
    a = np.array([js(t)["val_macro_auroc"] for t in tags(m)])
    y, p = val(tags(m)[0])[0], [val(t)[1] for t in tags(m)]
    allp += p
    L.append(f"| {m} | {' '.join(f'{v:.4f}' for v in a)} | {a.mean():.4f} +- {a.std(ddof=1):.4f} | {macro_auroc(y, np.mean(p, 0)):.4f} |")
    if m == "resnet1d_wang":
        floor = a.std(ddof=1)

L += ["", f"## 2. Seed-0 variants of resnet1d_wang (fold 9), noise floor = seed std {floor:.4f}", "",
      "| variant | fold-9 mAUROC | vs base | same run, best epoch |", "|---|---|---|---|"]
base = js(VARIANTS[0][1])["val_macro_auroc"]
for n, t in VARIANTS:
    r = js(t)
    be = f"{r['val_best_epoch_auroc']:.4f} (ep {r['best_epoch']})" if "val_best_epoch_auroc" in r else f"(ep {r['best_epoch']})"
    L.append(f"| {n} | {r['val_macro_auroc']:.4f} | {r['val_macro_auroc'] - base:+.4f} | {be} |")

L += ["", "## 3. Pre-specified ensemble (fold 9)", "",
      f"Mean of probabilities, seeds 0-4 of {', '.join(BACKBONES)} ({len(allp)} models): **{macro_auroc(y, np.mean(allp, 0)):.4f}**"]

if "--final" in sys.argv:
    def test(n):
        return np.load(os.path.join(D, "probs", f"{n}_test.npz"))
    L += ["", "## 4. Fold 10 (each entry evaluated once)", "",
          "| entry | members | fold 9 | fold 10 | 95% CI (patient bootstrap) |", "|---|---|---|---|---|"]
    for n in FINALS + [b[1] for b in BASELINES]:
        r = js(n)
        lo, hi = r["test_ci95_patient_bootstrap"]
        L.append(f"| {n} | {len(r['tags'])} | {r['val_macro_auroc']:.4f} | **{r['test_macro_auroc']:.4f}** | {lo:.4f} - {hi:.4f} |")
    L += ["", "Paired patient bootstrap (1000 resamples, same patients for both), fold-10 macro-AUROC difference:", "",
          "| ours | baseline | ours - baseline | 95% CI | P(diff <= 0) | baseline published |", "|---|---|---|---|---|---|"]
    for n in FINALS:
        za = test(n)
        for bn, bt, pub in BASELINES:
            zb = test(bt)
            assert np.array_equal(za["test_ecg_id"], zb["test_ecg_id"])
            if js(n)["tags"] == js(bt)["tags"]:
                L.append(f"| {n} | {bn} | same model | - | - | {pub:.3f} |")
                continue
            d, lo, hi, pv = paired_patient_bootstrap(za["y_test"], za["test"], zb["test"], za["test_patient_id"])
            L.append(f"| {n} | {bn} | {d:+.4f} | {lo:+.4f} to {hi:+.4f} | {pv:.3f} | {pub:.3f} |")
    r = js("FINAL2_ensemble")
    L += ["", "Fold-10 scores of the ensemble members (printed by evaluation/test_fold_eval.py, not used for any choice):", "",
          "| model | per seed (0-4) | mean +- std |", "|---|---|---|"]
    sc = {x["tag"]: x["test"] for x in r["runs"]}
    for m in BACKBONES:
        a = np.array([sc[t] for t in tags(m)])
        L.append(f"| {m} | {' '.join(f'{v:.4f}' for v in a)} | {a.mean():.4f} +- {a.std(ddof=1):.4f} |")
    L += ["", "Published (Strodthoff et al. 2021): resnet1d_wang .930, xresnet1d101 .928, inception1d .921, ensemble .934."]
open(os.path.join(D, "REPORT_seeds.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
