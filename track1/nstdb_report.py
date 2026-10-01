"""NSTDB real-noise benchmark report (RULES2.md section 1). Reads results/track2/grid/nstdb/*.npz (fold 9).
Writes results/track2/nstdb.{md,json}. Fold 10 is not read."""
from __future__ import annotations

import json
import os

import numpy as np

import nstdb
from corruptions import MODES, cond_name
from t2lib import D, GATED, REGIMES, SEEDS, Grid, f4, fmt_boot, mean_std, paired, snr_of, spearman

REAL = {"B0-aug-real": [f"B0augreal_s{s}" for s in SEEDS], "F-real": [f"Freal_s{s}" for s in SEEDS]}
g = Grid("nstdb", {**REGIMES, **REAL})
M = list(g.regimes)
MAIN = [k for k in M if k not in REAL]
FAMS, SNRS = nstdb.FAMILIES, nstdb.SNRS
cs = lambda fams, snrs: [cond_name(f, s, m) for f in fams for s in snrs for m in MODES]
G = {f: cs([f], SNRS) for f in FAMS}
G["nstdb_all"] = cs(FAMS, SNRS)
for s in SNRS:
    G.update({f"{f}@{s:+.0f}": cs([f], [s]) for f in FAMS})
    G[f"nstdb_all@{s:+.0f}"] = cs(FAMS, [s])
ens = {k: {grp: g.score(g.ens[k], c) for grp, c in {**G, "clean": ["clean"]}.items()} for k in M}
seed = {k: {grp: [g.score(P, c) for P in g.seed[k]] for grp, c in {"clean": ["clean"], "nstdb_all": G["nstdb_all"],
                                                                    "nstdb_all@-6": G["nstdb_all@-6"]}.items()} for k in M}
BG = {"nstdb_all@-6": G["nstdb_all@-6"], "nstdb_mixed@-6": G["nstdb_mixed@-6"]}
bs = g.boot({k: g.ens[k] for k in M}, BG)
PAIRS = [("F", "B0-aug"), ("E", "B0-aug"), ("D", "B0-aug"), ("C", "B0-aug"), ("B0-aug", "B0-clean")]
REAL_PAIRS = [p for p in [("B0-aug-real", "B0-aug"), ("F-real", "B0-aug"), ("F-real", "B0-aug-real"), ("F-real", "F")] if p[0] in M]
boot = {grp: {f"{a} - {b}": paired(bs[grp], {k: ens[k][grp] for k in M}, a, b) for a, b in PAIRS + REAL_PAIRS} for grp in BG}
fb = boot["nstdb_all@-6"]["F - B0-aug"]
claim = fb["lo"] > 0 and fb["diff"] >= 0.005

# r diagnostics
rd = {}
for k in [k for k in GATED + ["F-real"] if k in g.rens]:
    R = g.rens[k]
    allc = G["nstdb_all"]
    snr = np.concatenate([np.full(len(g.Y), snr_of(c)) for c in allc])
    rd[k] = {"clean": float(R[g.ix["clean"]].mean()),
             "by_condition": {c: float(R[g.ix[c]].mean()) for c in allc},
             "spearman_all": spearman(snr, np.concatenate([R[g.ix[c]] for c in allc])),
             "spearman_by_family": {f: spearman(np.concatenate([np.full(len(g.Y), snr_of(c)) for c in G[f]]),
                                                np.concatenate([R[g.ix[c]] for c in G[f]])) for f in FAMS},
             "spearman_with_clean": None}
    # with clean as +inf dB (rank-based, so any value above 0 dB works): r should be highest on clean
    cc = ["clean"] + allc
    rd[k]["spearman_with_clean"] = spearman(np.concatenate([np.full(len(g.Y), 99.0 if c == "clean" else snr_of(c)) for c in cc]),
                                            np.concatenate([R[g.ix[c]] for c in cc]))
syn = json.load(open(os.path.join(D, "groups.json")))["ensemble"]
json.dump({"ensemble": ens, "per_seed": seed, "boot": boot, "claim_F_beats_B0aug_real_noise": claim, "r": rd},
          open(os.path.join(D, "nstdb.json"), "w"), indent=1)

# ---------------------------------------------------------------- report
noise = nstdb.load()
L = ["# NSTDB real-noise benchmark (fold 9)", "",
     "Rules: RULES2.md section 1 (committed before this run). Descriptive; it does not change the Track 2 verdict. "
     "Fold 10 is not used here.", "",
     "## Setup", "",
     "- MIT-BIH NSTDB records bw, ma, em: 2 channels, 360 Hz, 650,000 samples (30.09 min) each; "
     f"resampled to 100 Hz (polyphase 5/18) -> {next(iter(noise.values())).shape[-1]} samples, then band-passed 0.5-40 Hz "
     "with the ECG filter.",
     f"- Time split per record (100 Hz samples): TRAIN {nstdb.split_ranges(180556)['train']}, 10 s gap, "
     f"EVAL {nstdb.split_ranges(180556)['eval']} (asserted disjoint; tests/test_nstdb.py). Only EVAL noise is used here.",
     "- Per 10 s fold-9 record: one random excerpt; each of the 12 leads takes one of the 2 noise channels at random; "
     "each lead's noise is scaled to the target SNR against that lead's power. nstdb_mixed = two distinct families, each "
     "unit power, summed and rescaled. Modes whole (10 s) and burst (2-4 s). Noise is added before dataset-level "
     "standardization. Fixed seeded RNG per (condition, record).",
     "- Models: existing checkpoints, seeds 0-2, 3-seed probability average unless stated. Group = mean macro-AUROC over "
     "its conditions; per-family groups pool SNR {0, -6} dB and both modes.", "",
     "## Scores per family (3-seed average; drop vs the model's own clean score in brackets)", "",
     "| model | clean | " + " | ".join(FAMS) + " | nstdb_all |", "|---" * (len(FAMS) + 3) + "|"]
row = lambda k, grps: " | ".join(f"{f4(ens[k][x])} ({ens[k]['clean'] - ens[k][x]:+.3f})" for x in grps)
for k in M:
    L.append(f"| {k}{' (trained on real noise)' if k in REAL else ''} | {f4(ens[k]['clean'])} | " + row(k, FAMS + ["nstdb_all"]) + " |")
for s in SNRS:
    L += ["", f"At {s:+.0f} dB only (both modes):", "", "| model | " + " | ".join(FAMS) + " | nstdb_all |", "|---" * (len(FAMS) + 2) + "|"]
    for k in M:
        L.append(f"| {k} | " + row(k, [f"{f}@{s:+.0f}" for f in FAMS] + [f"nstdb_all@{s:+.0f}"]) + " |")
L += ["", "Per condition (3-seed average macro-AUROC):", "",
      "| model | " + " | ".join(f"{f.replace('nstdb_', '')} {s:+.0f} {m}" for f in FAMS for s in SNRS for m in MODES) + " |",
      "|---" * (1 + len(FAMS) * len(SNRS) * len(MODES)) + "|"]
for k in M:
    L.append(f"| {k} | " + " | ".join(f4(g.score(g.ens[k], [cond_name(f, s, m)])) for f in FAMS for s in SNRS for m in MODES) + " |")
L += ["", "Per seed (0-2), mean +- std:", "", "| model | clean | nstdb_all | nstdb_all at -6 dB |", "|---|---|---|---|"]
for k in M:
    L.append(f"| {k} | " + " | ".join(mean_std(seed[k][x]) for x in ["clean", "nstdb_all", "nstdb_all@-6"]) + " |")
L += ["", "For reference, synthetic benchmark at -6 dB (REPORT.md, 3-seed average): " +
      ", ".join(f"{k} all_corrupted {f4(syn[k]['all_corrupted'])} / mixed {f4(syn[k]['mixed'])}" for k in MAIN if k in syn) + ".", "",
      "## Paired patient bootstrap at -6 dB (1000 resamples, seed 0)", "",
      "| comparison | nstdb_all @ -6 dB | nstdb_mixed @ -6 dB |", "|---|---|---|"]
for a, b in PAIRS:
    L.append(f"| {a} - {b} | " + " | ".join(fmt_boot(boot[grp][f"{a} - {b}"]) for grp in BG) + " |")
if REAL_PAIRS:
    L += ["", "Trained on real noise (separate rows; NSTDB TRAIN noise was in their augmentation, so these families are not unseen for them):", "",
          "| comparison | nstdb_all @ -6 dB | nstdb_mixed @ -6 dB |", "|---|---|---|"]
    for a, b in REAL_PAIRS:
        L.append(f"| {a} - {b} | " + " | ".join(fmt_boot(boot[grp][f"{a} - {b}"]) for grp in BG) + " |")
L += ["", f"Rule (descriptive, RULES2.md 1): claim \"F beats B0-aug on real noise\" only if on nstdb_all at -6 dB the CI lower "
      f"bound > 0 and the difference >= +0.005. Observed {fmt_boot(fb)} -> **{'claim holds' if claim else 'claim NOT supported'}**.",
      f"(nstdb_mixed at -6 dB, reported only: {fmt_boot(boot['nstdb_mixed@-6']['F - B0-aug'])}.)", "",
      "## r diagnostics (3-seed mean r per record)", "",
      "| model | r clean | r nstdb_all at 0 dB | r nstdb_all at -6 dB | Spearman(SNR, r), 0 and -6 dB | Spearman incl. clean | "
      + " | ".join(f"Spearman {f}" for f in FAMS) + " |", "|---" * (6 + len(FAMS)) + "|"]
for k, r in rd.items():
    m0 = np.mean([r["by_condition"][c] for c in G["nstdb_all@+0"]])
    m6 = np.mean([r["by_condition"][c] for c in G["nstdb_all@-6"]])
    L.append(f"| {k} | {f4(r['clean'])} | {f4(m0)} | {f4(m6)} | {r['spearman_all']:+.3f} | {r['spearman_with_clean']:+.3f} | "
             + " | ".join(f"{r['spearman_by_family'][f]:+.3f}" for f in FAMS) + " |")
L += ["", "Spearman(SNR, r) pairs each record's r with the condition SNR over all 16 NSTDB conditions (positive = r rises with SNR). "
      "'incl. clean' adds the clean records as the highest SNR level.", "",
      "Mean r by family x SNR x mode:", "",
      "| model | clean | " + " | ".join(f"{f.replace('nstdb_', '')} {s:+.0f} {m}" for f in FAMS for s in SNRS for m in MODES) + " |",
      "|---" * (2 + len(FAMS) * len(SNRS) * len(MODES)) + "|"]
for k, r in rd.items():
    L.append(f"| {k} | {f4(r['clean'])} | " + " | ".join(f4(r["by_condition"][cond_name(f, s, m)]) for f in FAMS for s in SNRS for m in MODES) + " |")
open(os.path.join(D, "nstdb.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
