"""Revision ablation report (RULES3.md sections 1-2; fold 9, descriptive).

Reads grid/ablation and grid/ablation_nstdb (B0-aug and F linked from grid/main and grid/nstdb) and writes
results/track2/revision_ablation.json and .md.
"""
from __future__ import annotations

import json
import os

import numpy as np

from t2lib import D, SEEDS, Grid, fast_macro_auroc, paired, snr_of, spearman

ABL = {"B0-aug": "B0aug", "F": "F", "F-m0.02": "Fm002", "F-m0.10": "Fm010", "F-wsev0.3": "Fws03",
       "F-wcons0": "Fwc0", "F-h128": "Fh128", "T": "T", "TF": "TF", "B0-aug-all": "B0augall", "F-all": "Fall"}
REG = {k: [f"{p}_s{s}" for s in SEEDS] for k, p in ABL.items()}
FAM = {"seen": ["baseline_wander", "emg"], "unseen": ["motion_burst", "dropout", "powerline"], "mixed": ["mixed"]}
GATED = [k for k in ABL if k not in ("B0-aug", "B0-aug-all")]


def main():
    gs, gn = Grid("ablation", REG), Grid("ablation_nstdb", REG)
    assert set(gs.regimes) == set(REG) and set(gn.regimes) == set(REG), (set(REG) - set(gs.regimes))
    at6 = lambda g, fams: [c for c in g.names if c != "clean" and c.split("|")[0] in fams and snr_of(c) == -6.0]
    groups = {f"{k}@-6": at6(gs, f) for k, f in FAM.items()}
    groups["all_corrupted@-6"] = at6(gs, sum(FAM.values(), []))
    ngroups = {"nstdb_all@-6": [c for c in gn.names if c != "clean" and snr_of(c) == -6.0]}
    out = {"scores": {}, "per_seed_mixed": {}, "paired": {}, "r": {}}
    for m in REG:
        out["scores"][m] = {"clean": gs.score(gs.ens[m], ["clean"]),
                            **{g: gs.score(gs.ens[m], cs) for g, cs in groups.items()},
                            **{g: gn.score(gn.ens[m], cs) for g, cs in ngroups.items()}}
        v = [gs.score(P, groups["mixed@-6"]) for P in gs.seed[m]]
        out["per_seed_mixed"][m] = [float(np.mean(v)), float(np.std(v, ddof=1))]
    bs = gs.boot(gs.ens, {g: groups[g] for g in ["mixed@-6", "unseen@-6"]})
    bn = gn.boot(gn.ens, ngroups)
    allb = {**bs, **bn}
    full = lambda g: {m: out["scores"][m][g] for m in REG}
    pairs = [(m, "B0-aug") for m in REG if m != "B0-aug"] + [(m, "F") for m in GATED if m != "F"] + [("F-all", "B0-aug-all")] * ("F-all" in REG)
    for a, b in pairs:
        out["paired"][f"{a} - {b}"] = {g: paired(allb[g], full(g), a, b) for g in allb}
    for m in GATED:
        R = gs.rens[m]
        cor = [c for c in gs.names if c != "clean"]
        snr = np.repeat([snr_of(c) for c in cor], len(gs.Y))
        r6 = np.mean([R[gs.ix[c]].mean() for c in groups["all_corrupted@-6"]])
        norm = (gs.Y[:, 0] == 1)
        out["r"][m] = {"r_clean": float(R[0].mean()), "r_-6": float(r6), "drop": float(R[0].mean() - r6),
                       "spearman": spearman(snr, np.stack([R[gs.ix[c]] for c in cor]).ravel()),
                       "r_norm": float(R[0][norm].mean()), "r_abnormal": float(R[0][~norm].mean())}
    json.dump(out, open(os.path.join(D, "revision_ablation.json"), "w"), indent=1)

    fb = lambda d: f"{d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
    md = ["# Revision ablations (RULES3.md; fold 9, descriptive)\n", "3-seed probability averages; groups at -6 dB, both modes.\n",
          "| model | clean | seen | unseen | mixed | all corrupted | NSTDB all | mixed per seed (mean +- sd) |", "|---" * 8 + "|"]
    for m in REG:
        s = out["scores"][m]
        md.append(f"| {m} | {s['clean']:.4f} | {s['seen@-6']:.4f} | {s['unseen@-6']:.4f} | {s['mixed@-6']:.4f} | "
                  f"{s['all_corrupted@-6']:.4f} | {s['nstdb_all@-6']:.4f} | {out['per_seed_mixed'][m][0]:.4f} +- {out['per_seed_mixed'][m][1]:.4f} |")
    md += ["\n## Paired patient bootstrap (1000 resamples, seed 0)\n", "| pair | mixed@-6 | unseen@-6 | nstdb_all@-6 |", "|---|---|---|---|"]
    for k, v in out["paired"].items():
        md.append(f"| {k} | {fb(v['mixed@-6'])} | {fb(v['unseen@-6'])} | {fb(v['nstdb_all@-6'])} |")
    md += ["\n## r diagnostics (3-seed mean r)\n", "| model | r clean | r -6 dB | drop | Spearman(SNR, r) | r NORM | r abnormal |",
           "|---" * 7 + "|"]
    for m, r in out["r"].items():
        md.append(f"| {m} | {r['r_clean']:.3f} | {r['r_-6']:.3f} | {r['drop']:+.3f} | {r['spearman']:+.3f} | "
                  f"{r['r_norm']:.3f} | {r['r_abnormal']:.3f} |")
    open(os.path.join(D, "revision_ablation.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
