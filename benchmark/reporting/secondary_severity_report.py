"""Secondary severity tables at 0 dB and +6 dB (RULES2.md section 2). Descriptive only, NOT part of the
Track 2 decision rule. Stored fold-9 predictions (grid/main), no training, no inference.
Writes results/robustness/secondary_severity.{md,json} and figures/group_vs_snr.png."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from noise.synthetic import ALL_CONDITION_FAMILIES, MODES, SEEN, UNSEEN, cond_name  # noqa: E402
from reporting.grid_utils import COLORS, D, GATED, MARKERS, Grid, f4, fmt_boot, mean_std, paired, snr_of, spearman  # noqa: E402

ALL_SNRS = [15.0, 6.0, 0.0, -6.0]
TABLE_SNRS = [0.0, 6.0]
COLS = ["seen_families", "unseen_families", "mixed", "all_corrupted"]


def groups(snr):
    c = lambda fams: [cond_name(f, snr, m) for f in fams for m in MODES]
    return {"seen_families": c(SEEN), "unseen_families": c(UNSEEN), "mixed": c(["mixed"]),
            "all_corrupted": c(SEEN + UNSEEN + ["mixed"])}


g = Grid("main")
M = list(g.regimes)
corr4 = [cond_name(f, s, m) for f in ALL_CONDITION_FAMILIES for s in ALL_SNRS for m in MODES]
sp = {k: spearman(np.concatenate([np.full(len(g.Y), snr_of(c)) for c in corr4]),
                  np.concatenate([g.rens[k][g.ix[c]] for c in corr4])) for k in GATED}
out = {"spearman_snr_r_4levels": sp}
L = ["# Secondary severity tables (fold 9, 0 dB and +6 dB) - descriptive", "",
     "Rules: RULES2.md section 2. These tables are descriptive and are **not** part of the Track 2 decision "
     "rule (RULES.md); the fold-9 verdict (NO-GO, severity set {-6 dB}) is unchanged.",
     "Stored fold-9 predictions from grid/main (the same seeded corruptions as REPORT.md); no training, no new inference.",
     "Group score = unweighted mean of macro-AUROC over the group's conditions at that SNR, both modes "
     "(seen_families = baseline_wander, emg; unseen_families = motion_burst, dropout, powerline; mixed; "
     "all_corrupted = all six families).", ""]
clean = {k: g.score(g.ens[k], ["clean"]) for k in M}
clean_seed = {k: [g.score(P, ["clean"]) for P in g.seed[k]] for k in M}
for snr in TABLE_SNRS:
    G = groups(snr)
    ens = {k: {grp: g.score(g.ens[k], cs) for grp, cs in G.items()} for k in M}
    seed = {k: {grp: [g.score(P, cs) for P in g.seed[k]] for grp, cs in G.items()} for k in M}
    bs = g.boot(g.ens, G)
    boot = {grp: {k: paired(bs[grp], {m: ens[m][grp] for m in M}, k, "B0-aug") for k in M if k != "B0-aug"} for grp in G}
    allc = G["all_corrupted"]
    rr = {k: {"clean": float(g.rens[k][g.ix["clean"]].mean()), "at_snr": float(np.mean([g.rens[k][g.ix[c]].mean() for c in allc]))}
          for k in GATED}
    out[f"{snr:+.0f}"] = {"ensemble": ens, "per_seed": seed, "boot_vs_B0aug": boot, "r": rr}
    L += [f"## {snr:+.0f} dB", "", "3-seed probability average; drop vs the model's own clean score in brackets.", "",
          "| model | clean | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 2) + "|"]
    for k in M:
        L.append(f"| {k} | {f4(clean[k])} | " + " | ".join(f"{f4(ens[k][c])} ({clean[k] - ens[k][c]:+.3f})" for c in COLS) + " |")
    L += ["", "Per seed (0-2), mean +- std:", "", "| model | clean | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 2) + "|"]
    for k in M:
        L.append(f"| {k} | {mean_std(clean_seed[k])} | " + " | ".join(mean_std(seed[k][c]) for c in COLS) + " |")
    L += ["", "Paired patient bootstrap vs B0-aug (1000 resamples, seed 0): difference [95% CI].", "",
          "| model | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for k in M:
        if k != "B0-aug":
            L.append(f"| {k} | " + " | ".join(fmt_boot(boot[c][k]) for c in COLS) + " |")
    L += ["", f"r (3-seed mean r per record): clean vs {snr:+.0f} dB (all families, both modes); Spearman(SNR, r) over the 4 levels.", "",
          f"| model | r clean | r at {snr:+.0f} dB | difference | Spearman(SNR, r), 4 levels |", "|---|---|---|---|---|"]
    for k in GATED:
        L.append(f"| {k} | {f4(rr[k]['clean'])} | {f4(rr[k]['at_snr'])} | {rr[k]['at_snr'] - rr[k]['clean']:+.4f} | {sp[k]:+.3f} |")
    L.append("")

# conditions that actually cause drops
L += ["## Drop vs clean by condition (3-seed average)", "",
      "Cells: clean macro-AUROC minus the condition's macro-AUROC. Drops >= 0.01 in bold.", ""]
for k in ["B0-clean", "B0-aug", "F"]:
    L += [f"**{k}** (clean {f4(clean[k])})", "", "| family | mode | +15 dB | +6 dB | 0 dB | -6 dB |", "|---|---|---|---|---|---|"]
    for f in ALL_CONDITION_FAMILIES:
        for m in MODES:
            cells = []
            for s in ALL_SNRS:
                d = clean[k] - g.score(g.ens[k], [cond_name(f, s, m)])
                cells.append(f"**{d:+.4f}**" if d >= 0.01 else f"{d:+.4f}")
            L.append(f"| {f} | {m} | " + " | ".join(cells) + " |")
    L.append("")
L += ["Mean drop over the six families by mode (3-seed average):", "",
      "| model | mode | +15 dB | +6 dB | 0 dB | -6 dB |", "|---|---|---|---|---|---|"]
bym = {}
for k in M:
    for m in MODES:
        v = [clean[k] - g.score(g.ens[k], [cond_name(f, s, m) for f in ALL_CONDITION_FAMILIES]) for s in ALL_SNRS]
        bym[f"{k}|{m}"] = v
        L.append(f"| {k} | {m} | " + " | ".join(f"{x:+.4f}" for x in v) + " |")
out["drop_by_mode"] = bym

# figure: group score vs SNR
FIG_M = ["B0-clean", "B0-aug", "D", "E", "F"]
xs = ALL_SNRS[::-1]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
for ax, grp in zip(axes, COLS):
    for k in FIG_M:
        v = [g.score(g.ens[k], groups(s)[grp]) for s in xs] + [clean[k]]
        ax.plot(range(len(xs) + 1), v, color=COLORS[k], marker=MARKERS[k], ms=6, lw=2, label=k)
    ax.set_xticks(range(len(xs) + 1), [f"{s:+.0f}" for s in xs] + ["clean"])
    ax.set_title(grp, fontsize=10)
    ax.set_xlabel("SNR (dB)", fontsize=9, color="#52514e")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("group macro-AUROC (both modes)", fontsize=9, color="#52514e")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
os.makedirs(os.path.join(D, "figures"), exist_ok=True)
fig.savefig(os.path.join(D, "figures", "group_vs_snr.png"), dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
L += ["", "Figure: figures/group_vs_snr.png - group score vs SNR (3-seed average) for B0-clean, B0-aug, D, E, F.", ""]
json.dump(out, open(os.path.join(D, "secondary_severity.json"), "w"), indent=1)
open(os.path.join(D, "secondary_severity.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
