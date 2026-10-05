"""Revision analyses of stored fold-9 predictions (RULES3.md section 3; descriptive, no training, no fold 10).

a. per-class AUROC under noise (+ paired bootstrap F - B0-aug per class)
b. calibration under noise: ECE / Brier vs SNR, raw and 2-fold patient cross-fitted Platt (fit on clean)
c. diagnostic signal in r and the residual r-noise relation after removing each record's clean r
Writes results/robustness/revision_analyses.json and .md.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from loaders.ptbxl import SUPERCLASSES
from evaluation.metrics import ece_binary
from reporting.grid_utils import D, GATED, REGIMES, SEEDS, Grid, auroc, snr_of, spearman

REAL = {**REGIMES, "B0-aug-real": [f"B0augreal_s{s}" for s in SEEDS], "F-real": [f"Freal_s{s}" for s in SEEDS]}
SYN_FAM = {"seen": ["baseline_wander", "emg"], "unseen": ["motion_burst", "dropout", "powerline"], "mixed": ["mixed"]}


def conds_at(names, fams, snr):
    return [c for c in names if c != "clean" and c.split("|")[0] in fams and snr_of(c) == snr]


def per_class(Y, P, conds, ix, idx=None):
    """(5,) AUROC per class, averaged over conds."""
    sl = slice(None) if idx is None else idx
    return np.mean([[auroc(Y[sl, k], P[ix[c]][sl, k]) for k in range(Y.shape[1])] for c in conds], 0)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def crossfit_platt(P, Y, PID, seed=0):
    """Per-class Platt fitted on CLEAN records (condition 0) of one patient half, applied to the other half."""
    rng = np.random.default_rng(seed)
    pids = np.unique(PID)
    half = np.isin(PID, rng.permutation(pids)[: len(pids) // 2])
    out = np.zeros_like(P)
    for test in (half, ~half):
        for k in range(Y.shape[1]):
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(P[0, ~test, k])[:, None], Y[~test, k])
            out[:, test, k] = lr.predict_proba(logit(P[:, test, k]).reshape(-1, 1))[:, 1].reshape(P.shape[0], -1)
    return out


def calib(Y, Pc):
    return (float(np.mean([ece_binary(Y[:, k], Pc[:, k]) for k in range(Y.shape[1])])),
            float(np.mean((Pc - Y) ** 2)))


def main():
    out, md = {}, ["# Revision analyses of stored fold-9 predictions (RULES3.md 3, descriptive)\n",
                   "3-seed probability averages from grid/main and grid/nstdb; fold 10 not touched.\n"]
    gm, gn = Grid("main"), Grid("nstdb", REAL)
    groups = {f"{g}@-6": (gm, conds_at(gm.names, f, -6.0)) for g, f in SYN_FAM.items()}
    groups["nstdb_all@-6"] = (gn, [c for c in gn.names if c != "clean" and snr_of(c) == -6.0])

    # a. per-class AUROC
    out["per_class"] = {}
    md.append("## a. Per-class AUROC under noise (3-seed average)\n")
    for gname, (g, cs) in [("clean", (gm, ["clean"]))] + list(groups.items()):
        md += [f"\n**{gname}**\n", "| model | " + " | ".join(SUPERCLASSES) + " | macro |", "|---" * 7 + "|"]
        out["per_class"][gname] = {}
        for m in g.regimes:
            v = per_class(g.Y, g.ens[m], cs, g.ix)
            out["per_class"][gname][m] = v.tolist()
            md.append(f"| {m} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {v.mean():.4f} |")
    md.append("\n**Paired patient bootstrap, F - B0-aug per class (1000 resamples, seed 0)**\n")
    md += ["| group | " + " | ".join(SUPERCLASSES) + " |", "|---" * 6 + "|"]
    out["per_class_F_minus_B0aug"] = {}
    for gname in ["mixed@-6", "nstdb_all@-6"]:
        g, cs = groups[gname]
        full = per_class(g.Y, g.ens["F"], cs, g.ix) - per_class(g.Y, g.ens["B0-aug"], cs, g.ix)
        bs = np.array([per_class(g.Y, g.ens["F"], cs, g.ix, i) - per_class(g.Y, g.ens["B0-aug"], cs, g.ix, i)
                       for i in g.resamples(1000, 0)])
        lo, hi = np.quantile(bs, .025, 0), np.quantile(bs, .975, 0)
        out["per_class_F_minus_B0aug"][gname] = {"diff": full.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
        md.append(f"| {gname} | " + " | ".join(f"{d:+.4f} [{a:+.4f}, {b:+.4f}]" for d, a, b in zip(full, lo, hi)) + " |")

    # b. calibration vs SNR
    md.append("\n## b. Calibration under noise (mean over classes; ECE 15 bins)\n")
    md.append("Platt = per-class Platt fitted on clean records, 2-fold patient cross-fitting (RULES3.md 3b).\n")
    out["calibration"] = {}
    for fam, g, snrs in [("mixed|whole", gm, [15.0, 6.0, 0.0, -6.0]), ("nstdb_mixed|whole", gn, [0.0, -6.0])]:
        f, mode = fam.split("|")
        cs = ["clean"] + [f"{f}|{s:+.0f}|{mode}" for s in snrs]
        md += [f"\n**{fam}** (ECE raw / ECE Platt / Brier raw / Brier Platt)\n",
               "| model | " + " | ".join(cs) + " |", "|---" * (len(cs) + 1) + "|"]
        out["calibration"][fam] = {}
        for m in g.regimes:
            Pp = crossfit_platt(g.ens[m], g.Y, g.PID)
            row = {}
            for c in cs:
                (er, br), (ep, bp) = calib(g.Y, g.ens[m][g.ix[c]]), calib(g.Y, Pp[g.ix[c]])
                row[c] = {"ece_raw": er, "ece_platt": ep, "brier_raw": br, "brier_platt": bp}
            out["calibration"][fam][m] = row
            md.append(f"| {m} | " + " | ".join(
                f"{r['ece_raw']:.3f} / {r['ece_platt']:.3f} / {r['brier_raw']:.3f} / {r['brier_platt']:.3f}"
                for r in row.values()) + " |")

    # c. diagnostic signal in r
    md.append("\n## c. Diagnostic signal in r (clean fold 9, 3-seed mean r)\n")
    md.append("AUROC of r alone for each label (>0.5: r higher in positives); 'LR-CV' = 5-fold patient-grouped "
              "logistic regression of the label from r (out-of-fold AUROC). Delta-r Spearman = Spearman(SNR, "
              "r(cond) - r(clean, same record)) over all corrupted (record, condition) pairs at the 4 SNRs.\n")
    md += ["| model | non-NORM vs NORM | " + " | ".join(SUPERCLASSES) + " | LR-CV (mean of 5) | Spearman(SNR, r) | "
           "Spearman(SNR, delta r) | same, NORM-only | same, non-NORM |", "|---" * 11 + "|"]
    out["r_diag"] = {}
    cor = [c for c in gm.names if c != "clean"]
    snr = np.repeat([snr_of(c) for c in cor], len(gm.Y))
    norm_only = (gm.Y[:, 0] == 1) & (gm.Y[:, 1:].sum(1) == 0)
    non_norm = gm.Y[:, 0] == 0
    for m in GATED:
        R = gm.rens[m]
        r0 = R[0]
        a_ab = auroc(non_norm, r0)
        a_k = [auroc(gm.Y[:, k], r0) for k in range(5)]
        cv = []
        for k in range(5):
            oof = np.zeros(len(r0))
            for tr, te in GroupKFold(5).split(r0, groups=gm.PID):
                oof[te] = LogisticRegression().fit(r0[tr, None], gm.Y[tr, k]).predict_proba(r0[te, None])[:, 1]
            cv.append(auroc(gm.Y[:, k], oof))
        Rc = np.stack([R[gm.ix[c]] for c in cor])
        dR = Rc - r0[None]
        sp_raw = spearman(snr, Rc.ravel())
        sp_d = spearman(snr, dR.ravel())
        strat = [spearman(np.repeat([snr_of(c) for c in cor], s.sum()), dR[:, s].ravel()) for s in (norm_only, non_norm)]
        out["r_diag"][m] = {"auroc_nonnorm": a_ab, "auroc_per_class": a_k, "lr_cv_auroc": cv,
                            "spearman_snr_r": sp_raw, "spearman_snr_delta_r": sp_d,
                            "spearman_snr_delta_r_norm_only": strat[0], "spearman_snr_delta_r_non_norm": strat[1]}
        md.append(f"| {m} | {a_ab:.3f} | " + " | ".join(f"{x:.3f}" for x in a_k) +
                  f" | {np.mean(cv):.3f} | {sp_raw:+.3f} | {sp_d:+.3f} | {strat[0]:+.3f} | {strat[1]:+.3f} |")

    json.dump(out, open(os.path.join(D, "revision_analyses.json"), "w"), indent=1)
    open(os.path.join(D, "revision_analyses.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
