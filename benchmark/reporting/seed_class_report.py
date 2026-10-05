"""Fold-9 follow-ups from stored predictions (grid/main), descriptive:
1. F vs B0-aug per seed and mean +- std per group at -6 dB, next to the 3-seed probability average.
2. r by clean superclass-only group (NORM, MI, STTC, CD, HYP), NORM vs non-NORM, and Spearman(SNR, r)
   inside NORM-only and non-NORM records (D, E, F; 4 SNR levels, all corrupted conditions).
Writes results/robustness/seed_and_class.{md,json}."""
from __future__ import annotations

import json
import os

import numpy as np

from noise.synthetic import ALL_CONDITION_FAMILIES, MODES, SEEN, UNSEEN, cond_name
from loaders.ptbxl import SUPERCLASSES
from reporting.grid_utils import D, GATED, Grid, f4, mean_std, snr_of, spearman

g = Grid("main")
SNR = -6.0
c = lambda fams: [cond_name(f, SNR, m) for f in fams for m in MODES]
G = {"clean": ["clean"], "seen_families": c(SEEN), "unseen_families": c(UNSEEN), "mixed": c(["mixed"]),
     "all_corrupted": c(SEEN + UNSEEN + ["mixed"])}
COLS = list(G)
out = {"per_seed": {}, "ensemble": {}}
L = ["# Per-seed spread and r by diagnostic class (fold 9, descriptive)", "",
     "Stored fold-9 predictions (grid/main). Not part of any decision rule.", "",
     "## 1. F vs B0-aug per seed at -6 dB", "",
     "Group score = mean macro-AUROC over the group's conditions at -6 dB, both modes. "
     "'3-seed avg' = macro-AUROC of the averaged probabilities (the score used by the rules).", "",
     "| model | seed | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 2) + "|"]
for k in ["B0-aug", "F"]:
    ps = [{grp: g.score(P, cs) for grp, cs in G.items()} for P in g.seed[k]]
    ens = {grp: g.score(g.ens[k], cs) for grp, cs in G.items()}
    out["per_seed"][k], out["ensemble"][k] = ps, ens
    for s, v in enumerate(ps):
        L.append(f"| {k} | {s} | " + " | ".join(f4(v[x]) for x in COLS) + " |")
    L.append(f"| {k} | mean +- std | " + " | ".join(mean_std([v[x] for v in ps]) for x in COLS) + " |")
    L.append(f"| {k} | **3-seed avg** | " + " | ".join(f"**{f4(ens[x])}**" for x in COLS) + " |")
L += ["", "F - B0-aug:", "", "| | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
for s in range(3):
    L.append(f"| seed {s} (F_s{s} - B0aug_s{s}) | " + " | ".join(
        f"{out['per_seed']['F'][s][x] - out['per_seed']['B0-aug'][s][x]:+.4f}" for x in COLS) + " |")
L.append("| mean of per-seed differences | " + " | ".join(
    f"{np.mean([out['per_seed']['F'][s][x] - out['per_seed']['B0-aug'][s][x] for s in range(3)]):+.4f}" for x in COLS) + " |")
L.append("| 3-seed avg difference | " + " | ".join(f"{out['ensemble']['F'][x] - out['ensemble']['B0-aug'][x]:+.4f}" for x in COLS) + " |")
L.append("| ensembling gain, B0-aug (avg - mean of seeds) | " + " | ".join(
    f"{out['ensemble']['B0-aug'][x] - np.mean([v[x] for v in out['per_seed']['B0-aug']]):+.4f}" for x in COLS) + " |")
L.append("| ensembling gain, F | " + " | ".join(
    f"{out['ensemble']['F'][x] - np.mean([v[x] for v in out['per_seed']['F']]):+.4f}" for x in COLS) + " |")

# ---------------------------------------------------------------- 2. r by class
Y = g.Y
only = {cl: (Y[:, i] == 1) & (Y.sum(1) == 1) for i, cl in enumerate(SUPERCLASSES)}
norm_only, non_norm = only["NORM"], Y[:, 0] == 0
corr4 = [cond_name(f, s, m) for f in ALL_CONDITION_FAMILIES for s in [15.0, 6.0, 0.0, -6.0] for m in MODES]
L += ["", "## 2. r by diagnostic class (clean fold-9 records, 3-seed mean r)", "",
      "Superclass-only = records whose only superclass label is that class. non-NORM = records without the NORM label "
      "(any combination of MI, STTC, CD, HYP).", "",
      "| model | " + " | ".join(f"{cl}-only (n={only[cl].sum()})" for cl in SUPERCLASSES) +
      f" | NORM (n={(Y[:, 0] == 1).sum()}) | non-NORM (n={non_norm.sum()}) |", "|---" * (len(SUPERCLASSES) + 3) + "|"]
out["r_by_class"] = {}
for k in GATED:
    r0 = g.rens[k][g.ix["clean"]]
    v = {cl: float(r0[m].mean()) for cl, m in only.items()}
    v.update(NORM_any=float(r0[Y[:, 0] == 1].mean()), non_NORM=float(r0[non_norm].mean()))
    out["r_by_class"][k] = v
    L.append(f"| {k} | " + " | ".join(f4(v[cl]) for cl in SUPERCLASSES) + f" | {f4(v['NORM_any'])} | {f4(v['non_NORM'])} |")
L += ["", "Spearman(SNR, r) over all (record, condition) pairs of the corrupted conditions at +15, +6, 0, -6 dB "
      "(all six families, both modes), computed inside each record subset:", "",
      f"| model | all records (n={len(Y)}) | NORM-only (n={norm_only.sum()}) | non-NORM (n={non_norm.sum()}) | "
      "mean r drop clean -> -6 dB, NORM-only | same, non-NORM |", "|---|---|---|---|---|---|"]
out["spearman_by_subset"] = {}
for k in ["D", "E", "F"]:
    R = g.rens[k]
    row = {}
    for nm, msk in [("all", np.ones(len(Y), bool)), ("NORM_only", norm_only), ("non_NORM", non_norm)]:
        row[nm] = spearman(np.concatenate([np.full(msk.sum(), snr_of(cc)) for cc in corr4]),
                           np.concatenate([R[g.ix[cc]][msk] for cc in corr4]))
        low = [cc for cc in corr4 if snr_of(cc) == -6.0]
        row[nm + "_drop"] = float(R[g.ix["clean"]][msk].mean() - np.mean([R[g.ix[cc]][msk].mean() for cc in low]))
    out["spearman_by_subset"][k] = row
    L.append(f"| {k} | {row['all']:+.3f} | {row['NORM_only']:+.3f} | {row['non_NORM']:+.3f} | "
             f"{row['NORM_only_drop']:+.4f} | {row['non_NORM_drop']:+.4f} |")
json.dump(out, open(os.path.join(D, "seed_and_class.json"), "w"), indent=1)
open(os.path.join(D, "seed_and_class.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
