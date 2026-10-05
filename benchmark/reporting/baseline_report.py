"""Fold-9 table of all crop runs + fold-10 results (final_eval outputs) vs published references.
Also scores fold-9 ensembles of the backbone runs (val probabilities only). Writes results/ptbxl_baselines/REPORT.md."""
import json
import os
from itertools import combinations

import numpy as np

from evaluation.metrics import macro_auroc

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "ptbxl_baselines")
REFS = [("xresnet1d101", .928), ("resnet1d_wang", .930), ("inception1d", .921), ("ensemble", .934)]
BACKBONES = ["M1_crop", "resnet1d_wang_crop", "xresnet1d50_crop", "xresnet1d101_crop", "inception1d_crop",
             "M1_crop_datasetnorm", "resnet1d_wang_crop_dsnorm", "xresnet1d50_crop_dsnorm",
             "xresnet1d101_crop_dsnorm", "inception1d_crop_dsnorm"]


def load(tag):
    p = os.path.join(D, f"{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


runs = sorted(f[:-5] for f in os.listdir(D) if f.endswith(".json") and "train_time_s" in open(os.path.join(D, f)).read())
L = ["# PTB-XL Track 1 - crops, backbones, ablations (seed 0)", "",
     "Selection on fold 9 only. Fold 10 is evaluated only via evaluation/test_fold_eval.py (table 3).", "",
     "## 1. All runs (fold 9)", "",
     "| run | model | fold-9 mAUROC (mean agg) | fold-9 (max agg) | params | best ep / run | train time |",
     "|---|---|---|---|---|---|---|"]
for t in runs:
    r = load(t)
    alt = r.get("val_macro_auroc_agg_max", float("nan"))
    L.append(f"| {t} | {r['model']} | {r['val_macro_auroc']:.4f} | {alt:.4f} | {r['n_params']:,} | "
             f"{r['best_epoch']} / {r['epochs_run']} | {r['train_time_s'] / 60:.1f} min |")

L += ["", "## 2. Fold-9 ensembles of backbone runs (mean of probabilities)", "", "| members | fold-9 mAUROC |", "|---|---|"]
P = {t: np.load(os.path.join(D, "probs", f"{t}.npz")) for t in BACKBONES if os.path.exists(os.path.join(D, "probs", f"{t}.npz"))}
ens = []
for k in range(2, len(P) + 1):
    for c in combinations(P, k):
        ens.append((macro_auroc(P[c[0]]["y_val"], np.mean([P[t]["val"] for t in c], 0)), c))
for v, c in sorted(ens, reverse=True)[:8]:
    L.append(f"| {' + '.join(c)} | {v:.4f} |")

L += ["", "## 3. Fold 10 (evaluated once per entry) vs published", "", "| model | fold-9 | fold-10 mAUROC | 95% CI (patient bootstrap) |", "|---|---|---|---|"]
for f in sorted(os.listdir(D)):
    if f.endswith(".json") and '"test_ci95_patient_bootstrap"' in open(os.path.join(D, f)).read():
        r = json.load(open(os.path.join(D, f)))
        lo, hi = r["test_ci95_patient_bootstrap"]
        L.append(f"| **{r['name']}** ({' + '.join(r['tags'])}) | {r['val_macro_auroc']:.4f} | "
                 f"**{r['test_macro_auroc']:.4f}** | {lo:.4f} - {hi:.4f} |")
L.append("| M1 full-length, no crops (previous run) | 0.8994 | 0.8967 | - |")
for n, v in REFS:
    L.append(f"| {n} (Strodthoff et al. 2021, published) | - | {v:.3f} | - |")
open(os.path.join(D, "REPORT.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
