"""BUT QDB report (RULES2.md section 5). Subject-level bootstrap (15 subjects, 1000 resamples, seed 0).
Reads results/track2/butqdb/*. Writes results/track2/butqdb.{md,json}."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from sl_infer import GATED, SL
from t2lib import D, auroc, spearman

B = os.path.join(D, "butqdb")
z = np.load(os.path.join(B, "windows.npz"))
log = pd.read_csv(os.path.join(B, "windows_log.csv"), dtype={"record": str, "subject": str})
cls, subj, motion = z["cls"], z["subject"], z["acc_rms"]
SUBJ = np.unique(subj)
MINW = 5
r_avg = lambda norm, k: np.mean([np.load(os.path.join(B, norm, f"{t}.npz"))["r"] for t in SL[k]], 0)
# value per window; "noise" orientation: +1 = higher value means worse quality (H1, H2), -1 for r
SCORES = {**{f"r {k}": (r_avg("pooled", k), -1) for k in GATED},
          **{f"r {k} (window z-score)": (r_avg("window", k), -1) for k in GATED},
          "H1": (z["H1"], 1), "H2": (z["H2"], 1)}
idx_of = {s: np.flatnonzero(subj == s) for s in SUBJ}
rng = np.random.default_rng(0)
RES = [rng.choice(SUBJ, len(SUBJ), replace=True) for _ in range(1000)]
PAIRS = [(1, 2), (2, 3), (1, 3)]


def within_auc(v, sgn):
    """{pair: {subject: AUROC}} for subjects with >= MINW windows in both classes; positive = higher class."""
    out = {}
    for a, b in PAIRS:
        out[f"{a}v{b}"] = {}
        for s in SUBJ:
            i = idx_of[s]
            ma, mb = cls[i] == a, cls[i] == b
            if ma.sum() >= MINW and mb.sum() >= MINW:
                sel = ma | mb
                out[f"{a}v{b}"][s] = auroc((cls[i][sel] == b).astype(float), sgn * v[i][sel])
    return out


def stats(v, idx):
    c = cls[idx]
    x = v[idx]
    m = {q: float(x[c == q].mean()) if (c == q).any() else np.nan for q in (1, 2, 3)}
    out = {"mean1": m[1], "mean2": m[2], "mean3": m[3], "d12": m[1] - m[2], "d23": m[2] - m[3], "d13": m[1] - m[3],
           "sp_class": spearman(c, x), "sp_motion": spearman(motion[idx], x)}
    for q in (1, 2, 3):
        out[f"sp_motion_c{q}"] = spearman(motion[idx][c == q], x[c == q]) if (c == q).sum() > 2 else np.nan
    return out


report = {}
for name, (v, sgn) in SCORES.items():
    full = stats(v, np.arange(len(v)))
    wa = within_auc(v, sgn)
    full.update({f"auc{p}": float(np.mean(list(d.values()))) for p, d in wa.items()})
    boots = {k: [] for k in full}
    for draw in RES:
        idx = np.concatenate([idx_of[s] for s in draw])
        st = stats(v, idx)
        for p, d in wa.items():
            vals = [d[s] for s in draw if s in d]
            st[f"auc{p}"] = float(np.mean(vals)) if vals else np.nan
        for k in boots:
            boots[k].append(st[k])
    ci = {k: (float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))) for k, b in boots.items()}
    # ordering count among subjects with >= MINW windows in all 3 classes
    elig = [s for s in SUBJ if all((cls[idx_of[s]] == q).sum() >= MINW for q in (1, 2, 3))]
    means = {s: [float(v[idx_of[s]][cls[idx_of[s]] == q].mean()) for q in (1, 2, 3)] for s in elig}
    ok = [s for s in elig if (means[s][0] > means[s][1] > means[s][2] if sgn < 0 else means[s][0] < means[s][1] < means[s][2])]
    report[name] = {"full": full, "ci": ci, "within_auc_by_subject": wa, "order_eligible": elig, "order_ok": ok,
                    "subject_means": means, "sign": sgn}
json.dump(report, open(os.path.join(D, "butqdb.json"), "w"), indent=1, default=float)


def excl0(name, key, null=0.0):
    lo, hi = report[name]["ci"][key]
    return lo > null or hi < null


def cell(name, key, null=0.0, fmt="{:+.3f}"):
    f, (lo, hi) = report[name]["full"][key], report[name]["ci"][key]
    s = f"{fmt.format(f)} [{fmt.format(lo)}, {fmt.format(hi)}]"
    return f"**{s}**" if excl0(name, key, null) else s


norm = np.load(os.path.join(B, "norm.npz"))
L = ["# BUT QDB - r and heuristics vs annotated ECG quality (no retraining, no tuning)", "",
     "Rules and assumptions: RULES2.md section 5 (committed before any BUT QDB run).", "",
     "**Protocol caveat.** The Scientific Data descriptor (Smital et al. 2026) says it gives an evaluation protocol and "
     "code, but its full text was not accessible (login/subscription) and no official code repository was found. This "
     "analysis follows the PhysioNet README and the assumptions A1-A8 in RULES2.md, not the descriptor's protocol. Results "
     "are not comparable to published BUT QDB benchmarks.", "",
     "**Other caveats.** (1) With 15 subjects, a finding is claimed only if its subject-bootstrap 95% CI excludes 0 "
     "(AUROC: excludes 0.5); bold = CI excludes the null. (2) Class 2 may be dominated by baseline-drift noise, which the "
     "0.5-40 Hz band-pass partly removes before the models and heuristics see the signal. (3) Class 3 is concentrated in "
     "one subject (105: 4,733 of the class-3 windows), so pooled class-3 statistics largely describe one person. "
     "(4) The models were trained on PTB-XL lead I; the BUT QDB lead (Bittium Faros 180, chest-worn) differs.", "",
     "## Data", "",
     f"- 18 recordings, {len(SUBJ)} subjects; ECG 1 channel at 1000 Hz, ACC 3 channels at 100 Hz (verified for every record; "
     "lengths in results/track2/butqdb/windows_log.csv). Labels: consensus columns.",
     f"- 10 s grid windows touching annotated samples: {int(log.grid_touching.sum())}; kept (one class throughout): "
     f"{int(log.kept.sum())}; dropped: {int(log.dropped.sum())} ({int(log.dropped_mixed.sum())} span two classes, "
     f"{int(log.dropped_partly_unannotated.sum())} partly unannotated).",
     f"- Kept windows per class: " + ", ".join(f"class {q} {int((cls == q).sum())}" for q in (1, 2, 3)) +
     f". Pooled standardization: mean {float(norm['mu']):.4f}, sd {float(norm['sd']):.4f} (mV).", "",
     "Windows per subject and class:", "", "| subject | records | class 1 | class 2 | class 3 | dropped |", "|---|---|---|---|---|---|"]
for s in SUBJ:
    g = log[log.subject == s]
    L.append(f"| {s} | {', '.join(g.record)} | {int(g.class1.sum())} | {int(g.class2.sum())} | {int(g.class3.sum())} | {int(g.dropped.sum())} |")
names = list(SCORES)
L += ["", "## a. Mean value per class and class differences (pooled windows; [95% subject-bootstrap CI])", "",
      "For r, class 1 > 2 > 3 is the expected direction (positive differences). For H1 and H2, the expected direction is negative differences.", "",
      "| score | class 1 | class 2 | class 3 | 1 - 2 | 2 - 3 | 1 - 3 | Spearman(class, value) |", "|---|---|---|---|---|---|---|---|"]
for n in names:
    f = report[n]["full"]
    L.append(f"| {n} | {f['mean1']:.4f} | {f['mean2']:.4f} | {f['mean3']:.4f} | {cell(n, 'd12', fmt='{:+.4f}')} | "
             f"{cell(n, 'd23', fmt='{:+.4f}')} | {cell(n, 'd13', fmt='{:+.4f}')} | {cell(n, 'sp_class')} |")
L += ["", f"Within-subject pairwise AUROC (subjects with >= {MINW} windows in both classes; score -r for r, the value for H1/H2; "
      "> 0.5 = worse quality scores as noisier), averaged over subjects:", "",
      "| score | 1v2 (n subj) | 2v3 (n subj) | 1v3 (n subj) |", "|---|---|---|---|"]
for n in names:
    wa = report[n]["within_auc_by_subject"]
    L.append(f"| {n} | " + " | ".join(f"{cell(n, f'auc{p}', 0.5, '{:.3f}')} ({len(wa[p])})" for p in ["1v2", "2v3", "1v3"]) + " |")
L += ["", "## b. Subjects whose per-class means follow the expected order", "",
      f"Eligible = >= {MINW} windows in each of the 3 classes. Order: r 1 > 2 > 3; H1, H2 1 < 2 < 3.", "",
      "| score | subjects in order / eligible | eligible subjects (class 1 / 2 / 3 means) |", "|---|---|---|"]
for n in names:
    rp = report[n]
    L.append(f"| {n} | {len(rp['order_ok'])} / {len(rp['order_eligible'])} | " +
             "; ".join(f"{s}{'*' if s in rp['order_ok'] else ''}: " + " / ".join(f"{x:.3f}" for x in rp["subject_means"][s]) for s in rp["order_eligible"]) + " |")
L += ["", "(* = in the expected order.)", "", "## c-d. Motion (ACC magnitude, high-pass 0.5 Hz, RMS per window) vs score", "",
      "| score | Spearman pooled | within class 1 | within class 2 | within class 3 |", "|---|---|---|---|---|"]
for n in names:
    L.append(f"| {n} | {cell(n, 'sp_motion')} | {cell(n, 'sp_motion_c1')} | {cell(n, 'sp_motion_c2')} | {cell(n, 'sp_motion_c3')} |")
L += ["", "For r, a negative Spearman with motion means r falls as motion rises (the expected direction); for H1/H2 a positive one.", "",
      "Motion by class (mean RMS): " + ", ".join(f"class {q} {motion[cls == q].mean():.2f}" for q in (1, 2, 3)) +
      f"; Spearman(class, motion) = {spearman(cls, motion):+.3f}.", ""]
open(os.path.join(D, "butqdb.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
