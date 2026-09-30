"""Track 2 fold-9 analysis: group scores, per-seed spread, r diagnostics, figures, paired bootstraps and
the GO / NO-GO verdict of RULES.md. Reads results/track2/grid/main/*.npz.
Writes results/track2/{groups.json, verdict.json, REPORT.md, figures/*.png}. Fold 10 is not read here."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from corruptions import ALL_CONDITION_FAMILIES, MODES, SEEN, UNSEEN, cond_name  # noqa: E402
from metrics import macro_auroc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "track2")
G = os.path.join(D, "grid", "main")
SEEDS = [0, 1, 2]
REGIMES = {"B0-clean": ["resnet1d_wang_crop_dsnorm", "resnet1d_wang_crop_dsnorm_s1", "resnet1d_wang_crop_dsnorm_s2"],
           **{k: [f"{p}_s{s}" for s in SEEDS] for k, p in
              [("B0-aug", "B0aug"), ("C", "C"), ("D", "D"), ("E", "E"), ("F", "F"), ("E-clean", "Eclean")]}}
GATED = ["D", "E", "F", "E-clean"]
BASELINES, CANDIDATES = ["B0-aug", "C", "D"], ["E", "F"]
COLORS = {"B0-clean": "#2a78d6", "B0-aug": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4",
          "F": "#008300", "E-clean": "#4a3aa7"}
MARKERS = {"B0-clean": "o", "B0-aug": "s", "C": "^", "D": "D", "E": "v", "F": "P", "E-clean": "X"}
SEV = json.load(open(os.path.join(D, "difficulty.json")))["severity_set"]
ALL_SNRS = [15.0, 6.0, 0.0, -6.0]


def groups(snrs):
    c = lambda fams: [cond_name(f, s, m) for f in fams for s in snrs for m in MODES]
    return {"seen_families": c(SEEN), "unseen_families": c(UNSEEN), "mixed": c(["mixed"]),
            "all_corrupted": c(SEEN + UNSEEN + ["mixed"])}


GROUPS = groups(SEV)

Z = {t: np.load(os.path.join(G, f"{t}.npz")) for ts in REGIMES.values() for t in ts}
z0 = Z[REGIMES["B0-clean"][0]]
NAMES, Y, PID = list(z0["conds"]), z0["y"], z0["patient_id"]
IX = {c: k for k, c in enumerate(NAMES)}
for t, z in Z.items():
    assert list(z["conds"]) == NAMES and np.array_equal(z["ecg_id"], z0["ecg_id"]), t
ENS = {k: np.mean([Z[t]["probs"] for t in ts], 0) for k, ts in REGIMES.items()}           # (C, N, 5)
RENS = {k: np.mean([Z[t]["r"] for t in ts], 0) for k in GATED for ts in [REGIMES[k]]}      # (C, N)


def score(P, conds, idx=None):
    sl = slice(None) if idx is None else idx
    return float(np.mean([macro_auroc(Y[sl], P[IX[c]][sl]) for c in conds]))


def table_scores(P):
    s = {"clean": score(P, ["clean"])}
    s.update({g: score(P, cs) for g, cs in GROUPS.items()})
    return s


ens_scores = {k: table_scores(P) for k, P in ENS.items()}
seed_scores = {k: [table_scores(Z[t]["probs"]) for t in ts] for k, ts in REGIMES.items()}
COLS = ["clean", "seen_families", "unseen_families", "mixed", "all_corrupted"]


def paired_group_bootstrap(PA, PB, conds, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    pids = np.unique(PID)
    idx_of = {p: np.flatnonzero(PID == p) for p in pids}
    d = []
    while len(d) < n:
        idx = np.concatenate([idx_of[p] for p in rng.choice(pids, len(pids), replace=True)])
        if (Y[idx].sum(0) == 0).any() or (Y[idx].sum(0) == len(idx)).any():
            continue
        d.append(score(PA, conds, idx) - score(PB, conds, idx))
    d = np.array(d)
    return {"diff": score(PA, conds) - score(PB, conds), "lo": float(np.quantile(d, .025)),
            "hi": float(np.quantile(d, .975)), "p_le_0": float((d <= 0).mean())}


# ---------------------------------------------------------------- r diagnostics
def r_diag(k):
    R = RENS[k]
    norm = Y[:, 0] == 1
    corr = [cond_name(f, s, m) for f in ALL_CONDITION_FAMILIES for s in ALL_SNRS for m in MODES]
    rs = np.concatenate([R[IX[c]] for c in corr])
    snr = np.concatenate([np.full(len(Y), float(c.split("|")[1])) for c in corr])
    low = [c for c in GROUPS["all_corrupted"] if float(c.split("|")[1]) == min(SEV)]
    return {"clean_mean_r": float(R[IX["clean"]].mean()),
            "lowest_chosen_snr_mean_r": float(np.mean([R[IX[c]].mean() for c in low])),
            "clean_norm_r": float(R[IX["clean"]][norm].mean()), "clean_abnormal_r": float(R[IX["clean"]][~norm].mean()),
            "spearman_snr_r_4levels": float(spearmanr(snr, rs).statistic),
            "by_condition": {c: float(R[IX[c]].mean()) for c in NAMES}}


RD = {k: r_diag(k) for k in GATED}

# ---------------------------------------------------------------- verdict
best_base = {g: max(BASELINES, key=lambda b: ens_scores[b][g]) for g in ["mixed", "unseen_families"]}
verdict = {"severity_set": SEV, "best_baseline": best_base, "candidates": {}}
for c in CANDIDATES:
    bm = paired_group_bootstrap(ENS[c], ENS[best_base["mixed"]], GROUPS["mixed"])
    bu = paired_group_bootstrap(ENS[c], ENS[best_base["unseen_families"]], GROUPS["unseen_families"])
    r = RD[c]
    checks = {
        "1_mixed": bm["lo"] > 0 and bm["diff"] >= 0.005,
        "2_unseen": bu["diff"] >= 0.003 and bu["lo"] > -0.002,
        "3_r_severity": (r["clean_mean_r"] - r["lowest_chosen_snr_mean_r"] >= 0.05) and r["spearman_snr_r_4levels"] >= 0.3,
        "4_r_abnormal": r["clean_abnormal_r"] >= r["clean_norm_r"] - 0.05,
        "5_clean": ens_scores[c]["clean"] >= ens_scores["B0-aug"]["clean"] - 0.003,
    }
    verdict["candidates"][c] = {"boot_mixed": bm, "boot_unseen": bu, "checks": checks, "go": all(checks.values()),
                                "clean_minus_B0aug": ens_scores[c]["clean"] - ens_scores["B0-aug"]["clean"]}
passing = [c for c in CANDIDATES if verdict["candidates"][c]["go"]]
verdict["winner"] = max(passing, key=lambda c: ens_scores[c]["mixed"]) if passing else None
verdict["verdict"] = "GO" if passing else "NO-GO"
json.dump({"ensemble": ens_scores, "per_seed": seed_scores, "r_diagnostics": RD}, open(os.path.join(D, "groups.json"), "w"), indent=1)
json.dump(verdict, open(os.path.join(D, "verdict.json"), "w"), indent=1)

# ---------------------------------------------------------------- figures
os.makedirs(os.path.join(D, "figures"), exist_ok=True)
XS = ALL_SNRS[::-1] + ["clean"]
XT = [f"{s:+.0f}" for s in ALL_SNRS[::-1]] + ["clean"]


def small_multiples(series, value, ylabel, fname, title):
    fig, axes = plt.subplots(2, len(ALL_CONDITION_FAMILIES), figsize=(17, 6.4), sharey=True, sharex=True)
    for i, m in enumerate(MODES):
        for j, f in enumerate(ALL_CONDITION_FAMILIES):
            ax = axes[i, j]
            for k in series:
                v = [value(k, "clean" if s == "clean" else cond_name(f, s, m)) for s in XS]
                ax.plot(range(len(XS)), v, color=COLORS[k], marker=MARKERS[k], ms=6, lw=2, label=k)
            ax.axvspan(-0.3, 0.3, color="#52514e", alpha=0.08, lw=0) if -6.0 in SEV else None
            ax.set_xticks(range(len(XS)), XT, fontsize=8)
            ax.grid(axis="y", color="#e6e5e0", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors="#52514e", labelsize=8)
            if i == 0:
                ax.set_title(f, fontsize=10, color="#0b0b0b")
            if j == 0:
                ax.set_ylabel(f"{m}\n{ylabel}", fontsize=9, color="#52514e")
            if i == 1:
                ax.set_xlabel("SNR (dB)", fontsize=8, color="#52514e")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(series), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.05, fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "figures", fname), dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


small_multiples(list(REGIMES), lambda k, c: macro_auroc(Y, ENS[k][IX[c]]), "macro-AUROC", "perf_vs_snr.png",
                "Fold 9, 3-seed probability average: macro-AUROC vs SNR per family (shaded = chosen severity)")
small_multiples(GATED, lambda k, c: RD[k]["by_condition"][c], "mean r", "r_vs_snr.png",
                "Fold 9: mean gate r (3-seed average, per record over windows) vs SNR per family")

# ---------------------------------------------------------------- report
f4 = lambda v: f"{v:.4f}"
L = ["# PTB-XL Track 2 - calibration, corruption benchmark, reliability-gated variants", "",
     "Rules: RULES.md (committed before any Track 2 run; severity set and amendment in section E).",
     "Fold-10 evaluations: FOLD10_LOG.md. Everything below the calibration section is fold 9 only.", "",
     "## 1. Calibration (resnet1d_wang_crop_dsnorm seeds 0-2, fitted on fold 9, scored on fold 10)", "",
     open(os.path.join(D, "calibration.md")).read(), open(os.path.join(D, "difficulty.md")).read().replace("## ", "## 2. ", 1),
     f"## 3. Group scores (fold 9, severity set {SEV} dB, both modes)", "",
     "3-seed probability average; drop vs the model's own clean score in brackets.", "",
     "| model | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
for k, s in ens_scores.items():
    L.append(f"| {k} | {f4(s['clean'])} | " + " | ".join(f"{f4(s[g])} ({s['clean'] - s[g]:+.3f})" for g in COLS[1:]) + " |")
L += ["", "Per seed (seeds 0-2), mean +- std:", "", "| model | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
for k, ss in seed_scores.items():
    L.append(f"| {k} | " + " | ".join(f"{np.mean([s[g] for s in ss]):.4f} +- {np.std([s[g] for s in ss], ddof=1):.4f}" for g in COLS) + " |")
L += ["", f"Per condition at {min(SEV):+.0f} dB (3-seed average):", "",
      "| model | " + " | ".join(f"{f} {m}" for f in ALL_CONDITION_FAMILIES for m in MODES) + " |",
      "|---" * (1 + 2 * len(ALL_CONDITION_FAMILIES)) + "|"]
for k in REGIMES:
    L.append(f"| {k} | " + " | ".join(f4(macro_auroc(Y, ENS[k][IX[cond_name(f, min(SEV), m)]])) for f in ALL_CONDITION_FAMILIES for m in MODES) + " |")
L += ["", "Figure: figures/perf_vs_snr.png (all four SNR levels, diagnostic only).", "",
      "## 4. r diagnostics (fold 9, 3-seed mean r per record)", "",
      "| model | clean | at -6 dB (all families, modes) | clean NORM | clean abnormal | Spearman(SNR, r), 4 levels |", "|---|---|---|---|---|---|"]
for k, r in RD.items():
    L.append(f"| {k} | {f4(r['clean_mean_r'])} | {f4(r['lowest_chosen_snr_mean_r'])} | {f4(r['clean_norm_r'])} | "
             f"{f4(r['clean_abnormal_r'])} | {r['spearman_snr_r_4levels']:+.3f} |")
for k in GATED:
    L += ["", f"Mean r by family x SNR x mode, {k} (clean {f4(RD[k]['clean_mean_r'])}):", "",
          "| family | mode | " + " | ".join(f"{s:+.0f} dB" for s in ALL_SNRS) + " |", "|---|---|" + "---|" * len(ALL_SNRS)]
    for f in ALL_CONDITION_FAMILIES:
        for m in MODES:
            L.append(f"| {f} | {m} | " + " | ".join(f4(RD[k]["by_condition"][cond_name(f, s, m)]) for s in ALL_SNRS) + " |")
L += ["", "Figure: figures/r_vs_snr.png.", "", "## 5. Verdict (fold 9)", "",
      f"Best baseline (3-seed average): mixed = {best_base['mixed']} ({f4(ens_scores[best_base['mixed']]['mixed'])}), "
      f"unseen_families = {best_base['unseen_families']} ({f4(ens_scores[best_base['unseen_families']]['unseen_families'])}).", "",
      "| rule | " + " | ".join(CANDIDATES) + " |", "|---" * (len(CANDIDATES) + 1) + "|"]
yn = lambda b: "pass" if b else "FAIL"
rows = [("1. mixed: diff, 95% CI (need lo > 0, diff >= +0.005)",
         lambda c: (lambda b: f"{b['diff']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] {yn(verdict['candidates'][c]['checks']['1_mixed'])}")(verdict["candidates"][c]["boot_mixed"])),
        ("2. unseen: diff, 95% CI (need diff >= +0.003, lo > -0.002)",
         lambda c: (lambda b: f"{b['diff']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] {yn(verdict['candidates'][c]['checks']['2_unseen'])}")(verdict["candidates"][c]["boot_unseen"])),
        ("3. r clean - r at -6 dB (need >= 0.05); Spearman (need >= +0.3)",
         lambda c: f"{RD[c]['clean_mean_r'] - RD[c]['lowest_chosen_snr_mean_r']:+.4f}; {RD[c]['spearman_snr_r_4levels']:+.3f} {yn(verdict['candidates'][c]['checks']['3_r_severity'])}"),
        ("4. r abnormal - r NORM (need >= -0.05)",
         lambda c: f"{RD[c]['clean_abnormal_r'] - RD[c]['clean_norm_r']:+.4f} {yn(verdict['candidates'][c]['checks']['4_r_abnormal'])}"),
        ("5. clean - B0-aug clean (need >= -0.003)",
         lambda c: f"{verdict['candidates'][c]['clean_minus_B0aug']:+.4f} {yn(verdict['candidates'][c]['checks']['5_clean'])}"),
        ("all rules", lambda c: "**GO**" if verdict["candidates"][c]["go"] else "**NO-GO**")]
for name, fn in rows:
    L.append(f"| {name} | " + " | ".join(fn(c) for c in CANDIDATES) + " |")
L += ["", f"**Fold-9 verdict: {verdict['verdict']}**" + (f" - winner {verdict['winner']}." if verdict["winner"] else
      " - no candidate passes all rules; fold 10 is not scored; findings go to the benchmark-and-analysis paper."), ""]
fin = os.path.join(D, "fold10.md")
if os.path.exists(fin):
    L += [open(fin).read()]
open(os.path.join(D, "REPORT.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
