"""RULES4.md report: settings S1-S4 (wang, xresnet, inception, chapman), Holm-corrected confirmatory tests, TOST,
effect sizes, DerSimonian-Laird pooling, gate clamp sweep, selective prediction and the ablation table.

Reads grid/r4_{setting}, grid/r4_{setting}_nstdb and grid/r4clamp_{setting}_r{R}; writes
results/track2/rules4_report.json and rules4_report.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

import nstdb
from corruptions import corrupt_split
from sl_infer import h1
from t2lib import D, Grid, auroc, fast_macro_auroc, snr_of, spearman
from train import filtered

SPLIT_SEED = 20261004
MARGIN = 0.005
ALPHA = 0.05
CLAMP = ["0", "0.25", "0.5", "0.75", "1"]
FAM = {"baseline_wander": ["baseline_wander"], "emg": ["emg"], "motion_burst": ["motion_burst"],
       "dropout": ["dropout"], "powerline": ["powerline"], "seen": ["baseline_wander", "emg"],
       "unseen": ["motion_burst", "dropout", "powerline"], "mixed": ["mixed"],
       "all_corrupted": ["baseline_wander", "emg", "motion_burst", "dropout", "powerline", "mixed"]}


def _t(p, seeds=(0, 1, 2)):
    return [f"{p}_s{s}" for s in seeds]


S5 = (0, 1, 2, 3, 4)
SETTINGS = {
    "wang": {"dataset": "ptbxl", "reg": {
        "Clean": ["resnet1d_wang_crop_dsnorm"] + [f"resnet1d_wang_crop_dsnorm_s{s}" for s in S5[1:]],
        "Aug": _t("B0aug", S5), "Gate": _t("D", S5), "DiffGate": _t("E", S5), "SevGate": _t("F", S5),
        "DiffGate-clean": _t("Eclean"), "Aug-real": _t("B0augreal"), "SevGate-real": _t("Freal"),
        "Clean (s0-2)": ["resnet1d_wang_crop_dsnorm", "resnet1d_wang_crop_dsnorm_s1", "resnet1d_wang_crop_dsnorm_s2"],
        "Aug (s0-2)": _t("B0aug"), "DiffGate (s0-2)": _t("E"), "SevGate (s0-2)": _t("F")}},
    **{n: {"dataset": "ptbxl", "reg": {
        "Clean": [f"{trunk}_crop_dsnorm", f"{trunk}_crop_dsnorm_s1", f"{trunk}_crop_dsnorm_s2"],
        "Aug": _t(f"{p}B0aug"), "DiffGate": _t(f"{p}E"), "SevGate": _t(f"{p}F"), "DiffGate-clean": _t(f"{p}Eclean"),
        "Aug-real": _t(f"{p}B0augreal"), "SevGate-real": _t(f"{p}Freal")}}
       for n, p, trunk in [("xresnet", "X", "xresnet1d50"), ("inception", "I", "inception1d")]},
    "chapman": {"dataset": "chapman", "reg": {
        "Clean": _t("CHB0clean"), "Aug": _t("CHB0aug"), "DiffGate": _t("CHE"), "SevGate": _t("CHF"),
        "DiffGate-clean": _t("CHEclean"), "Aug-real": _t("CHB0augreal"), "SevGate-real": _t("CHFreal")}},
}
GATES = ["DiffGate", "SevGate"]


def at6(g: Grid, fams) -> list[str]:
    return [c for c in g.names if c != "clean" and c.split("|")[0] in fams and snr_of(c) == -6.0]


def holm(p: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm step-down: reject in order of increasing p until the first p > alpha / (m - k)."""
    order, rej, m = sorted(p, key=p.get), {}, len(p)
    stop = False
    for k, key in enumerate(order):
        stop = stop or p[key] > alpha / (m - k)
        rej[key] = not stop
    return rej


def dersimonian_laird(y, se) -> dict:
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    w = 1 / v
    mu_f = (w * y).sum() / w.sum()
    q = (w * (y - mu_f) ** 2).sum()
    k = len(y)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = (ws * y).sum() / ws.sum()
    s = float(np.sqrt(1 / ws.sum()))
    return {"pooled": float(mu), "lo": float(mu - 1.96 * s), "hi": float(mu + 1.96 * s), "se": s, "tau2": float(tau2),
            "I2": float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0, "Q": float(q), "k": k}


def compare(bs: dict, full: dict, a: str, b: str) -> dict:
    d = bs[a] - bs[b]
    lo90, hi90 = np.quantile(d, [.05, .95])
    return {"diff": full[a] - full[b], "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975)),
            "lo90": float(lo90), "hi90": float(hi90), "se": float(d.std(ddof=1)), "p_one_sided": float((d <= 0).mean()),
            "equivalent": bool(-MARGIN < lo90 and hi90 < MARGIN)}


def setting_report(name: str, cfg: dict) -> dict:
    gs, gn = Grid(f"r4_{name}", cfg["reg"]), Grid(f"r4_{name}_nstdb", cfg["reg"])
    missing = [m for m in cfg["reg"] if m not in gs.regimes and m not in gn.regimes]
    groups = {f"{k}@-6": at6(gs, f) for k, f in FAM.items()}
    groups["clean"] = ["clean"]
    ng = {"nstdb_all@-6": [c for c in gn.names if c != "clean" and snr_of(c) == -6.0],
          **{f"{f}@-6": [c for c in gn.names if c.startswith(f + "|") and snr_of(c) == -6.0] for f in nstdb.FAMILIES}}
    out = {"n_records": int(len(gs.Y)), "missing": missing, "scores": {}, "per_seed": {}, "tests": {}, "dz": {},
           "share": {}, "r": {}}
    for m in gs.regimes:
        out["scores"][m] = {g: gs.score(gs.ens[m], cs) for g, cs in groups.items()}
        out["per_seed"][m] = {g: [gs.score(P, groups[g]) for P in gs.seed[m]] for g in ["clean", "mixed@-6", "unseen@-6"]}
    for m in gn.regimes:
        out["scores"].setdefault(m, {}).update({g: gn.score(gn.ens[m], cs) for g, cs in ng.items()})
        out["per_seed"].setdefault(m, {})["nstdb_all@-6"] = [gn.score(P, ng["nstdb_all@-6"]) for P in gn.seed[m]]
    main = [m for m in ["Clean", "Aug", *GATES, "Gate", "DiffGate-clean"] if m in gs.regimes]
    bs = gs.boot({m: gs.ens[m] for m in main}, {g: groups[g] for g in ["mixed@-6", "unseen@-6", "clean"]})
    nm = [m for m in ["Aug", *GATES, "Aug-real", "SevGate-real"] if m in gn.regimes]
    bn = gn.boot({m: gn.ens[m] for m in nm}, {"nstdb_all@-6": ng["nstdb_all@-6"]})
    allb = {**bs, **bn}
    pairs = [(gt, "Aug") for gt in GATES + ["Gate"]] + [("SevGate-real", "Aug-real")]
    for a, b in pairs:
        for g, B in allb.items():
            if a in B and b in B:
                out["tests"].setdefault(f"{a} - {b}", {})[g] = compare(
                    B, {m: out["scores"][m][g] for m in B}, a, b)
    for a, b in pairs:
        for g in ["mixed@-6", "unseen@-6", "nstdb_all@-6", "clean"]:
            if a in out["per_seed"] and b in out["per_seed"] and g in out["per_seed"][a]:
                k = min(len(out["per_seed"][a][g]), len(out["per_seed"][b][g]))
                d = np.array(out["per_seed"][a][g][:k]) - np.array(out["per_seed"][b][g][:k])
                out["dz"].setdefault(f"{a} - {b}", {})[g] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                                              "dz": float(d.mean() / d.std(ddof=1)), "n_seeds": k}
    for gt in GATES:
        for g in ["mixed@-6", "unseen@-6"]:
            s = out["scores"]
            out["share"].setdefault(gt, {})[g] = float((s[gt][g] - s["Aug"][g]) / (s["Aug"][g] - s["Clean"][g]))
    for m in gs.rens:
        R = gs.rens[m]
        cor = [c for c in gs.names if c != "clean"]
        snr = np.repeat([snr_of(c) for c in cor], len(gs.Y))
        r6 = np.mean([R[gs.ix[c]].mean() for c in groups["all_corrupted@-6"]])
        out["r"][m] = {"r_clean": float(R[0].mean()), "r_-6": float(r6), "drop": float(R[0].mean() - r6),
                       "spearman": spearman(snr, np.stack([R[gs.ix[c]] for c in cor]).ravel())}
    out["clamp"] = clamp(name, cfg, gs, gn, groups, ng)
    out["selective"] = selective(cfg, gs, gn)
    return out


def clamp(name, cfg, gs, gn, groups, ng) -> dict:
    res = {}
    for m in GATES:
        tags = cfg["reg"][m]
        row = {"learned": {"clean": gs.score(gs.ens[m], ["clean"]), "mixed@-6": gs.score(gs.ens[m], groups["mixed@-6"]),
                           "unseen@-6": gs.score(gs.ens[m], groups["unseen@-6"]),
                           "nstdb_all@-6": gn.score(gn.ens[m], ng["nstdb_all@-6"])}}
        for r in CLAMP:
            if not all(os.path.exists(os.path.join(D, "grid", f"r4clamp_{name}_r{r}", f"{t}.npz")) for t in tags):
                continue
            g = Grid(f"r4clamp_{name}_r{r}", {m: tags})
            if m not in g.regimes:
                continue
            P = g.ens[m]
            row[f"r={r}"] = {"clean": g.score(P, ["clean"]), "mixed@-6": g.score(P, at6(g, ["mixed"])),
                             "unseen@-6": g.score(P, at6(g, FAM["unseen"])),
                             "nstdb_all@-6": g.score(P, [c for c in g.names if c.startswith("nstdb_")])}
        consts = [k for k in row if k != "learned"]
        if consts:
            best = {c: max(row[k][c] for k in consts) for c in row["learned"]}
            row["learned_minus_best_constant"] = {c: row["learned"][c] - best[c] for c in best}
            row["constant_matches_learned"] = any(
                all(abs(row[k][c] - row["learned"][c]) <= 0.002 for c in row["learned"]) for k in consts)
        res[m] = row
    return res


def selective(cfg, gs, gn) -> dict:
    """RULES4.md 6: clean/corrupted 50/50 mixtures; detection AUROC and macro-AUROC by coverage."""
    d, _, _ = filtered(cfg["dataset"])
    Xf = d["test" if cfg["dataset"] == "chapman" else "val"][0]
    ids = gs.ids
    assert len(Xf) == len(ids)
    is_cor = np.array([np.random.default_rng([SPLIT_SEED, int(i)]).random() < 0.5 for i in ids])
    h1_of = lambda X: np.array([np.mean([h1(x[c]) for c in range(x.shape[0])]) for x in X])
    h_clean = h1_of(Xf)
    out = {}
    for label, g, cond, Xc_fn in [
            ("synthetic mixed|-6|whole", gs, "mixed|-6|whole", lambda: corrupt_split(Xf, ids, "mixed", -6.0, "whole")),
            ("nstdb_mixed|-6|whole", gn, "nstdb_mixed|-6|whole", lambda: nstdb.corrupt_split(Xf, ids, "nstdb_mixed", -6.0, "whole"))]:
        h = np.where(is_cor, h1_of(Xc_fn()), h_clean)
        k0, k1 = g.ix["clean"], g.ix[cond]
        res = {}
        for m in ["Aug", "Aug-real", "DiffGate", "SevGate", "SevGate-real"]:
            if m not in g.regimes:
                continue
            P = np.where(is_cor[:, None], g.ens[m][k1], g.ens[m][k0])
            q = np.clip(P, 1e-7, 1 - 1e-7)
            ent = -(q * np.log(q) + (1 - q) * np.log(1 - q)).mean(1)
            scores = {"entropy": ent, "H1": h}
            if m in g.rens:
                scores["-r"] = -np.where(is_cor, g.rens[m][k1], g.rens[m][k0])
            row = {"all": fast_macro_auroc(g.Y, P), "detect_auroc": {}, "coverage": {}}
            for sname, s in scores.items():
                row["detect_auroc"][sname] = auroc(is_cor, s)
                row["coverage"][sname] = {}
                for cov in [1.0, .9, .8, .7, .6, .5]:
                    keep = np.argsort(s, kind="stable")[:int(round(cov * len(s)))]
                    row["coverage"][sname][f"{int(cov * 100)}"] = fast_macro_auroc(g.Y[keep], P[keep])
            rng = np.random.default_rng(0)
            row["coverage"]["random"] = {f"{int(c * 100)}": float(np.mean([
                fast_macro_auroc(g.Y[ix], P[ix]) for ix in (rng.permutation(len(P))[:int(round(c * len(P)))] for _ in range(20))]))
                for c in [1.0, .9, .8, .7, .6, .5]}
            res[m] = row
        out[label] = res
    return out


def verdict(rejected: bool, t: dict) -> str:
    """RULES4.md 3: helps (Holm + practical bar); otherwise significance and TOST are reported side by side."""
    if rejected and t["diff"] >= MARGIN:
        return "helps"
    sig = "significant, below practical bar" if rejected else "not significant"
    return f"{sig}; " + ("equivalent" if t["equivalent"] else "not equivalent")


def main():
    rep = {n: setting_report(n, cfg) for n, cfg in SETTINGS.items()}
    fam = {f"{gt} - Aug, {n}": rep[n]["tests"][f"{gt} - Aug"]["mixed@-6"] for n in rep for gt in GATES}
    rej = holm({k: v["p_one_sided"] for k, v in fam.items()})
    confirm = {k: {**v, "holm_reject": rej[k], "helps": rej[k] and v["diff"] >= MARGIN,
                   "verdict": verdict(rej[k], v)} for k, v in fam.items()}
    meta = {}
    for gt in GATES:
        for g in ["mixed@-6", "unseen@-6", "nstdb_all@-6"]:
            ts = [rep[n]["tests"][f"{gt} - Aug"][g] for n in rep]
            meta.setdefault(gt, {})[g] = dersimonian_laird([t["diff"] for t in ts], [t["se"] for t in ts])
    out = {"settings": rep, "confirmatory": confirm, "meta": meta}
    json.dump(out, open(os.path.join(D, "rules4_report.json"), "w"), indent=1)
    md = write_md(out)
    open(os.path.join(D, "rules4_report.md"), "w").write(md)
    print(md)


def fb(t):
    return f"{t['diff']:+.4f} [{t['lo']:+.4f}, {t['hi']:+.4f}]"


def write_md(o) -> str:
    L = ["# RULES4 report (robustness of the conclusion)\n",
         "Ensembles: probability average over seeds (S1 wang: seeds 0-4 for Clean/Aug/Gate/DiffGate/SevGate). "
         "PTB-XL settings on fold 9; Chapman on its held-out test split. Groups at -6 dB, both modes.\n",
         "## Confirmatory tests (mixed@-6, Holm over 8 one-sided tests, alpha 0.05; TOST margin 0.005)\n",
         "| comparison | delta [95% CI] | 90% CI | one-sided p | Holm reject | verdict |", "|---|---|---|---|---|---|"]
    for k, v in o["confirmatory"].items():
        L.append(f"| {k} | {fb(v)} | [{v['lo90']:+.4f}, {v['hi90']:+.4f}] | {v['p_one_sided']:.3f} | {v['holm_reject']} | {v['verdict']} |")
    L += ["\n## Random-effects meta-analysis over S1-S4 (DerSimonian-Laird)\n",
          "| gate - Aug | endpoint | pooled [95% CI] | tau^2 | I^2 |", "|---|---|---|---|---|"]
    for gt, gs in o["meta"].items():
        for g, m in gs.items():
            L.append(f"| {gt} | {g} | {m['pooled']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] | {m['tau2']:.2e} | {m['I2']:.2f} |")
    for n, r in o["settings"].items():
        L += [f"\n## Setting {n} (N = {r['n_records']})" + (f" - missing: {r['missing']}" if r["missing"] else "") + "\n",
              "| model | clean | seen | unseen | mixed | all corr. | NSTDB all | bw | emg | motion | dropout | powerline |",
              "|---" * 12 + "|"]
        for m, s in r["scores"].items():
            f = lambda k: f"{s[k]:.4f}" if k in s else "-"
            L.append(f"| {m} | " + " | ".join(f(k) for k in ["clean", "seen@-6", "unseen@-6", "mixed@-6", "all_corrupted@-6",
                                                           "nstdb_all@-6", "baseline_wander@-6", "emg@-6", "motion_burst@-6",
                                                           "dropout@-6", "powerline@-6"]) + " |")
        L += ["\nPer seed (mean +- s.d.): " + "; ".join(
            f"{m} mixed@-6 {np.mean(v['mixed@-6']):.4f} +- {np.std(v['mixed@-6'], ddof=1):.4f}"
            for m, v in r["per_seed"].items() if "mixed@-6" in v and len(v["mixed@-6"]) > 1) + "\n",
              "| comparison | endpoint | delta [95% CI] | p (one-sided) | TOST equiv. | d_z (seeds) |", "|---|---|---|---|---|---|"]
        for k, gs in r["tests"].items():
            for g, t in gs.items():
                dz = r["dz"].get(k, {}).get(g)
                L.append(f"| {k} | {g} | {fb(t)} | {t['p_one_sided']:.3f} | {t['equivalent']} | "
                         + (f"{dz['dz']:+.2f} (n={dz['n_seeds']})" if dz else "-") + " |")
        L.append("\nShare of the augmentation gain added by the gate: " + "; ".join(
            f"{gt} {g} {v:+.2f}" for gt, gs in r["share"].items() for g, v in gs.items()))
        L += ["\nr diagnostics: " + "; ".join(f"{m}: clean {v['r_clean']:.3f}, -6 dB {v['r_-6']:.3f}, "
                                               f"Spearman(SNR, r) {v['spearman']:+.3f}" for m, v in r["r"].items())]
        L += ["\nClamp sweep (seed-averaged; r = 1 uses F_local only):\n",
              "| model | r | clean | mixed@-6 | unseen@-6 | NSTDB all@-6 |", "|---|---|---|---|---|---|"]
        for m, row in r["clamp"].items():
            for k, v in row.items():
                if isinstance(v, dict) and k != "learned_minus_best_constant":
                    L.append(f"| {m} | {k} | {v['clean']:.4f} | {v['mixed@-6']:.4f} | {v['unseen@-6']:.4f} | {v['nstdb_all@-6']:.4f} |")
            if "constant_matches_learned" in row:
                L.append(f"| {m} | learned - best const. | " + " | ".join(
                    f"{row['learned_minus_best_constant'][c]:+.4f}" for c in ["clean", "mixed@-6", "unseen@-6", "nstdb_all@-6"])
                         + f" |\n\n{m}: a constant r within 0.002 of learned on every condition: {row['constant_matches_learned']}\n")
        for mix, res in r["selective"].items():
            L += [f"\nSelective prediction, 50/50 clean / {mix}:\n",
                  "| model | score | detect AUROC | 100% | 90% | 80% | 70% | 60% | 50% |", "|---" * 9 + "|"]
            for m, row in res.items():
                for sname in [*row["detect_auroc"], "random"]:
                    det = f"{row['detect_auroc'][sname]:.3f}" if sname in row["detect_auroc"] else "-"
                    L.append(f"| {m} | {sname} | {det} | " + " | ".join(
                        f"{row['coverage'][sname][c]:.4f}" for c in ["100", "90", "80", "70", "60", "50"]) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
