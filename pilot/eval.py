"""Phase 1 evaluation (docs/master.md B6-B8, docs/CONTRACT.md).

    python eval.py --config configs/phase1.yaml [--runs runs/] [--smoke]

Evaluates every complete run (best.pt + meta.json) on PTB-XL fold 10 under a fixed,
precomputed set of corrupted test signals (identical for every model and seed) and writes
to results/: per_run.csv, per_run_groups.csv, table_mean_std.csv, bootstrap.csv,
reliability.csv, figures/*.png and summary.md (B8 verdict).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train import (REPO, SUPERCLASSES, auroc_per_class, get_device, load_config, macro_auroc,
                   macro_f1, predict)

CANDIDATES = ["E", "F"]                   # RACER-lite, RACER-lite + losses
BASELINES = ["A", "B", "C", "D", "G"]     # augmented controls (only those with runs are used)
R_VARIANTS = ["D", "E", "F"]              # models with a reliability/gate map
SEV_ORDER = ["mild", "moderate", "severe"]
SEV_LABEL = {"mild": "15 dB", "moderate": "6 dB", "severe": "0 dB"}
DROPOUT_LABEL = {"mild": "0.5 s", "moderate": "1 s", "severe": "2 s"}
BINDING_GROUPS = ["mixed"]  # default for eval.binding_groups (real_noise needs NSTDB, not in Phase 1)
SUPPORTING = [("mixed", "severe"), ("real_noise", "severe"), ("unseen_families", "all"),
              ("unseen_families", "severe"), ("unseen_no_em", "all"), ("unseen_no_em", "severe")]


# --------------------------------------------------------------------------- conditions
@dataclass(frozen=True)
class Condition:
    family: str      # corruption family, "mixed", or "clean"
    severity: str    # mild/moderate/severe, or "none" for clean
    mode: str        # whole/burst, "na" (dropout: mode-independent) or "none" (clean)

    @property
    def name(self) -> str:
        return "clean" if self.family == "clean" else f"{self.family}|{self.severity}|{self.mode}"


def build_conditions(all_families, severities, modes) -> list[Condition]:
    conds = [Condition("clean", "none", "none")]
    for fam in list(all_families) + ["mixed"]:
        for sev in severities:
            for mode in (["na"] if fam == "dropout" else modes):
                conds.append(Condition(fam, sev, mode))
    return conds


def mix_pool(all_families) -> list[str]:
    """Families eligible for the 'mixed' test condition: all additive-noise families.
    dropout is excluded because a 'mix' is defined as a sum of unit-power noises (CONTRACT)."""
    return [f for f in all_families if f != "dropout"]


def group_families(train_families, unseen_families) -> dict[str, list[str]]:
    """Condition groups over the configured families; groups that would be empty or identical
    to another (real_noise without NSTDB, unseen_no_em without nstdb_em) are left out."""
    fams = list(train_families) + list(unseen_families)
    g = {}
    real = [f for f in fams if f.startswith("nstdb_")]
    if real:
        g["real_noise"] = real
    g["seen_families"] = list(train_families)
    g["unseen_families"] = list(unseen_families)
    if "nstdb_em" in unseen_families:
        g["unseen_no_em"] = [f for f in unseen_families if f != "nstdb_em"]  # docs/master.md D4
    g["mixed"] = ["mixed"]
    return g


def group_members(conds: list[Condition], families: list[str], severity: str) -> list[str]:
    return [c.name for c in conds if c.family in families and (severity == "all" or c.severity == severity)]


def record_rng(noise_seed: int, cond: Condition, i: int) -> np.random.Generator:
    """Per-record generator keyed on (seed, family, mode, record) but NOT severity: the three
    severities of a family/mode share noise excerpt, channel and burst position and differ
    only in scale (common random numbers -> paired severity curves)."""
    key = zlib.crc32(f"{cond.family}|{cond.mode}".encode())
    return np.random.default_rng([noise_seed, key, i])


def corrupt_condition(x_raw: np.ndarray, cond: Condition, corruptor, noise_seed: int,
                      pool: list[str]) -> np.ndarray:
    """RAW corrupted test signals [N, T] (preprocessing is applied by the caller)."""
    if cond.family == "clean":
        return x_raw.copy()
    out = np.empty_like(x_raw)
    for i in range(x_raw.shape[0]):
        rng = record_rng(noise_seed, cond, i)
        fam = cond.family
        if fam == "mixed":
            pick = np.sort(rng.choice(len(pool), size=2, replace=False))
            fam = [pool[j] for j in pick]
        out[i] = corruptor.apply(x_raw[i], fam, cond.severity, cond.mode if cond.mode != "na" else "whole", rng)
    return out


def _file_digest(*paths: Path) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(p.read_bytes() if p.exists() else b"missing")
    return h.hexdigest()[:12]


class TestSetCache:
    """Corrupted + preprocessed test sets, computed once per condition and cached as .npy.
    The cache key covers the condition, noise seed, test ecg_ids, the mix pool and the source
    of data.py / corruptions.py, so any change there invalidates it."""

    def __init__(self, x_raw, ecg_ids, corruptor, preprocess, noise_seed, pool, cache_dir: Path,
                 fs: int = 100):
        self.x_raw, self.corruptor, self.preprocess = x_raw, corruptor, preprocess
        self.noise_seed, self.pool = noise_seed, pool
        self.dir = cache_dir / "eval_test"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.base = hashlib.sha1(np.asarray(ecg_ids, dtype=np.int64).tobytes()
                                 + str((noise_seed, pool, fs)).encode()
                                 + _file_digest(REPO / "data.py", REPO / "corruptions.py").encode()).hexdigest()[:12]

    def get(self, cond: Condition) -> np.ndarray:
        safe = cond.name.replace("|", "_")
        path = self.dir / f"{safe}_{self.base}.npy"
        if path.exists():
            return np.load(path)
        x = self.preprocess(corrupt_condition(self.x_raw, cond, self.corruptor, self.noise_seed, self.pool))
        x = np.asarray(x, dtype=np.float32)
        np.save(path, x)
        return x


# --------------------------------------------------------------------------- bootstrap
def patient_bootstrap_indices(patient_ids: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Cluster bootstrap: resample PATIENTS with replacement (as many as in the test set) and
    take all records of each drawn patient."""
    _, inv = np.unique(np.asarray(patient_ids), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    clusters = np.split(order, np.cumsum(counts)[:-1])
    n_pat = len(clusters)
    return [np.concatenate([clusters[k] for k in rng.integers(0, n_pat, n_pat)]) for _ in range(n_boot)]


def bootstrap_auroc(y: np.ndarray, probs: dict[str, dict[str, np.ndarray]], idx_list) -> dict:
    """Macro-AUROC per model and condition on each bootstrap resample (same resamples for every
    model and condition -> paired). Returns {model: {cond: array[n_boot]}}."""
    out = {m: {c: np.empty(len(idx_list)) for c in pc} for m, pc in probs.items()}
    for b, idx in enumerate(idx_list):
        yb = y[idx]
        for m, pc in probs.items():
            for c, p in pc.items():
                auc = auroc_per_class(yb, p[idx])
                out[m][c][b] = np.nanmean(auc) if np.isfinite(auc).any() else np.nan
    return out


def diff_ci(point_a, point_b, boot_a, boot_b, members, ci=0.95) -> tuple[float, float, float]:
    """Group difference = mean over member conditions of AUROC(a) - AUROC(b); percentile CI."""
    d = float(np.mean([point_a[c] - point_b[c] for c in members]))
    bd = np.mean([boot_a[c] - boot_b[c] for c in members], axis=0)
    a = (1 - ci) / 2
    lo, hi = np.nanpercentile(bd, [100 * a, 100 * (1 - a)])
    return d, float(lo), float(hi)


# --------------------------------------------------------------------------- decision rule
def apply_decision_rule(cands: dict[str, dict], tol: float = 0.05,
                        binding_groups: list[str] = BINDING_GROUPS) -> dict:
    """Mechanical B8 rule. cands[name] = {
         "ci": {group: (best_baseline, diff, lo, hi)} for every group in binding_groups,
         "supporting": {label: (best_baseline, diff, lo, hi)} (reported, non-binding),
         "r_sev": {"mild": float, "moderate": float, "severe": float},
         "r_abn": float, "r_norm": float}
    GO iff some candidate passes every binding criterion: CI of the macro-AUROC difference vs
    the best augmented baseline strictly above 0 for each binding group; mean r strictly
    decreasing mild > moderate > severe; r_abn >= r_norm - tol."""
    rows, passing, missing = [], [], []
    for name, c in cands.items():
        ok = True
        for g in binding_groups:
            v = c.get("ci", {}).get(g)
            if v is None or not np.isfinite(v[2]):
                missing.append(f"{name}:{g}")
                ok = False
                continue
            base, d, lo, hi = v
            p = lo > 0
            ok &= p
            rows.append(dict(candidate=name, criterion=f"AUROC diff vs best aug baseline ({base}) on {g}",
                             value=f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]", rule="95% CI lower bound > 0",
                             passed=p, binding=True))
        rs = c.get("r_sev")
        if rs is None or any(rs.get(s) is None or not np.isfinite(rs[s]) for s in SEV_ORDER):
            missing.append(f"{name}:r_sev")
            ok = False
        else:
            p = rs["mild"] > rs["moderate"] > rs["severe"]
            ok &= p
            rows.append(dict(candidate=name, criterion="mean r monotone in severity",
                             value=" > ".join(f"{rs[s]:.5f}" for s in SEV_ORDER),
                             rule="r(15 dB) > r(6 dB) > r(0 dB)", passed=p, binding=True))
        ra, rn = c.get("r_abn"), c.get("r_norm")
        if ra is None or rn is None or not (np.isfinite(ra) and np.isfinite(rn)):
            missing.append(f"{name}:r_clean")
            ok = False
        else:
            p = ra >= rn - tol
            ok &= p
            rows.append(dict(candidate=name, criterion="r on clean-abnormal vs clean-NORM",
                             value=f"{ra:.5f} vs {rn:.5f}", rule=f"r_abn >= r_norm - {tol}",
                             passed=p, binding=True))
        for label, v in c.get("supporting", {}).items():
            if v is None:
                continue
            base, d, lo, hi = v
            rows.append(dict(candidate=name, criterion=f"AUROC diff vs best aug baseline ({base}) on {label}",
                             value=f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]", rule="95% CI lower bound > 0",
                             passed=bool(lo > 0), binding=False))
        if ok:
            passing.append(name)
    if passing:
        verdict = "GO"
    elif not cands or (missing and not any(r["binding"] and not r["passed"] for r in rows)):
        verdict = "INCOMPLETE"  # nothing failed, but required numbers are missing
    else:
        verdict = "NO-GO / PIVOT"
    return {"verdict": verdict, "passing": passing, "missing": missing, "criteria": rows}


# --------------------------------------------------------------------------- run discovery
def discover_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for d in sorted(runs_dir.glob("*")):
        if (d / "best.pt").exists() and (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text())
            th = json.loads((d / "thresholds.json").read_text())
            runs.append({**meta, "dir": d, "thresholds": np.array([th[k] for k in SUPERCLASSES])})
    return runs


def load_run_model(run: dict, device):
    import models
    m = models.build_model(run["model"], n_classes=len(SUPERCLASSES), fs=int(run["fs"]))
    ckpt = torch.load(run["dir"] / "best.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(ckpt["model_state"])
    return m.to(device).eval()


# --------------------------------------------------------------------------- figures
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100",
          "E": "#e87ba4", "F": "#008300", "G": "#4a3aa7"}  # fixed categorical order
MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D", "E": "v", "F": "P", "G": "X"}


def plot_vs_severity(df: pd.DataFrame, value: str, families: list[str], variants: list[str],
                     regimes: list[str], ylabel: str, title: str, path: Path) -> None:
    """df columns: variant, regime, seed, family, severity, <value> (mode-averaged per seed).
    One panel per family; x = clean, mild, moderate, severe; mean +/- std over seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(families)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharey=True, squeeze=False)
    xs = np.arange(4)
    clean = df[df.family == "clean"]
    for ax, fam in zip(axes.flat, families):
        sub = df[df.family == fam]
        for v in variants:
            for reg in regimes:
                pts = []
                for sev in ["none"] + SEV_ORDER:
                    s = (clean if sev == "none" else sub)
                    s = s[(s.variant == v) & (s.regime == reg) & (s.severity == sev)][value]
                    pts.append((s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0) if len(s) else (np.nan, 0.0))
                if all(np.isnan(p[0]) for p in pts):
                    continue
                m, sd = np.array(pts).T
                ls = "-" if reg == "aug" else "--"
                ax.errorbar(xs, m, yerr=sd, color=COLORS.get(v, "gray"), marker=MARKERS.get(v, "o"),
                            ms=5, lw=1.8, ls=ls, capsize=2, label=f"{v} ({reg})")
        labels = DROPOUT_LABEL if fam == "dropout" else SEV_LABEL
        ax.set_xticks(xs, ["clean"] + [labels[s] for s in SEV_ORDER])
        ax.set_title(fam, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    h, lab = axes.flat[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 7), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def fmt_ms(m, s) -> str:
    return "n/a" if not np.isfinite(m) else (f"{m:.4f} ± {s:.4f}" if np.isfinite(s) else f"{m:.4f}")


def evaluate(cfg: dict, runs_dir: Path) -> dict:
    import data
    from corruptions import MODES, SEVERITIES, Corruptor
    from train import needs_nstdb

    P, D, E = cfg["paths"], cfg["data"], cfg["eval"]
    fs = int(D["fs"])
    TRAIN_FAMILIES = list(cfg["corruptions"]["train_families"])
    UNSEEN_FAMILIES = list(cfg["corruptions"]["unseen_families"])
    ALL_FAMILIES = TRAIN_FAMILIES + UNSEEN_FAMILIES
    binding = list(E.get("binding_groups", BINDING_GROUPS))
    out_dir = REPO / P["results_dir"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    runs = discover_runs(runs_dir)
    if not runs:
        raise SystemExit(f"no complete runs under {runs_dir}")
    bad_fs = sorted({r["dir"].name for r in runs if int(r["fs"]) != fs})
    if bad_fs:
        raise SystemExit(f"runs trained at a different fs than data.fs={fs}: {bad_fs}")
    print(f"{len(runs)} complete runs under {runs_dir}; device={device}")

    X, Y, meta = data.load_ptbxl(str(REPO / P["ptbxl_root"]), D["test_folds"], lead=D["lead"],
                                 cache_dir=str(REPO / P["cache_dir"]), fs=fs)
    pids = meta["patient_id"].to_numpy()
    nstdb = None
    if needs_nstdb(cfg):  # NSTDB TEST split only (last 40% in time)
        nstdb = data.load_nstdb(str(REPO / P["nstdb_root"]), target_fs=fs, train_frac=D["nstdb_train_frac"])
    corruptor = Corruptor(nstdb, "test", fs=fs, train_families=TRAIN_FAMILIES)
    conds = build_conditions(ALL_FAMILIES, SEVERITIES, MODES)
    pool = mix_pool(ALL_FAMILIES)
    cache = TestSetCache(X, meta["ecg_id"].to_numpy(), corruptor, lambda x: data.preprocess(x, fs),
                         E["noise_seed"], pool, REPO / P["cache_dir"], fs)
    cond_by_name = {c.name: c for c in conds}

    models_ = [(r, load_run_model(r, device)) for r in runs]
    key = lambda r: (r["variant"], r["regime"], int(r["seed"]))  # noqa: E731
    probs = defaultdict(dict)   # run key -> cond -> [N, K]
    rmean = defaultdict(dict)   # run key -> cond -> [N]
    rows = []
    for ci, cond in enumerate(conds):
        x = cache.get(cond)
        for r, m in models_:
            p, rr = predict(m, x, device, E["batch_size"])
            probs[key(r)][cond.name] = p
            if rr is not None:
                rmean[key(r)][cond.name] = rr
            rows.append(dict(variant=r["variant"], regime=r["regime"], seed=int(r["seed"]),
                             condition=cond.name, family=cond.family, severity=cond.severity,
                             mode=cond.mode, macro_auroc=macro_auroc(Y, p),
                             macro_f1=macro_f1(Y, p, r["thresholds"])))
        print(f"  [{ci + 1}/{len(conds)}] {cond.name}")
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "per_run.csv", index=False)

    # ---- group aggregates per run (unweighted mean over member conditions), drop = clean - group
    gfam = group_families(TRAIN_FAMILIES, UNSEEN_FAMILIES)
    gfam["all_corrupted"] = list(ALL_FAMILIES) + ["mixed"]
    grows = []
    for (v, reg, s), sub in per_run.groupby(["variant", "regime", "seed"]):
        sub = sub.set_index("condition")
        ca, cf = sub.loc["clean", "macro_auroc"], sub.loc["clean", "macro_f1"]
        grows.append(dict(variant=v, regime=reg, seed=s, group="clean", severity="none",
                          macro_auroc=ca, macro_f1=cf, drop_auroc=0.0, drop_f1=0.0))
        for g, fams in gfam.items():
            for sev in SEV_ORDER + ["all"]:
                mem = group_members(conds, fams, sev)
                if not mem:
                    continue
                a, f = sub.loc[mem, "macro_auroc"].mean(), sub.loc[mem, "macro_f1"].mean()
                grows.append(dict(variant=v, regime=reg, seed=s, group=g, severity=sev, macro_auroc=a,
                                  macro_f1=f, drop_auroc=ca - a, drop_f1=cf - f))
    groups = pd.DataFrame(grows)
    groups.to_csv(out_dir / "per_run_groups.csv", index=False)
    agg = (groups.groupby(["variant", "regime", "group", "severity"], sort=False)
           .agg(n_seeds=("seed", "nunique"),
                auroc_mean=("macro_auroc", "mean"), auroc_std=("macro_auroc", "std"),
                f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                drop_auroc_mean=("drop_auroc", "mean"), drop_auroc_std=("drop_auroc", "std"),
                drop_f1_mean=("drop_f1", "mean"), drop_f1_std=("drop_f1", "std"))
           .reset_index())
    agg.to_csv(out_dir / "table_mean_std.csv", index=False)
    cond_agg = (per_run.groupby(["variant", "regime", "condition", "family", "severity", "mode"], sort=False)
                .agg(n_seeds=("seed", "nunique"), auroc_mean=("macro_auroc", "mean"),
                     auroc_std=("macro_auroc", "std"), f1_mean=("macro_f1", "mean"),
                     f1_std=("macro_f1", "std")).reset_index())
    cond_agg.to_csv(out_dir / "table_conditions_mean_std.csv", index=False)

    # ---- paired patient-cluster bootstrap on seed-averaged probabilities (aug regime)
    aug_models = {}
    for v in CANDIDATES + BASELINES:
        ks = [k for k in probs if k[0] == v and k[1] == "aug"]
        if ks:
            aug_models[v] = {c.name: np.mean([probs[k][c.name] for k in ks], axis=0) for c in conds}
    point = {v: {c: macro_auroc(Y, p) for c, p in pc.items()} for v, pc in aug_models.items()}
    brng = np.random.default_rng(E["bootstrap_seed"])
    idx_list = patient_bootstrap_indices(pids, E["n_bootstrap"], brng)
    print(f"bootstrap: {len(idx_list)} patient-cluster resamples x {len(aug_models)} models x {len(conds)} conditions")
    boot = bootstrap_auroc(Y, aug_models, idx_list)
    brows = []
    best_base = {}
    for g in list(gfam):
        for sev in SEV_ORDER + ["all"]:
            mem = group_members(conds, gfam[g], sev)
            if not mem:
                continue
            bases = [b for b in BASELINES if b in aug_models]
            if bases:
                best_base[(g, sev)] = max(bases, key=lambda b: np.mean([point[b][c] for c in mem]))
            for cand in [c for c in CANDIDATES if c in aug_models]:
                for b in bases:
                    d, lo, hi = diff_ci(point[cand], point[b], boot[cand], boot[b], mem, E["ci"])
                    brows.append(dict(candidate=cand, baseline=b, group=g, severity=sev,
                                      n_conditions=len(mem),
                                      auroc_candidate=np.mean([point[cand][c] for c in mem]),
                                      auroc_baseline=np.mean([point[b][c] for c in mem]),
                                      diff=d, ci_low=lo, ci_high=hi, ci_excludes_zero=bool(lo > 0 or hi < 0),
                                      is_best_baseline=(b == best_base[(g, sev)])))
    bdf = pd.DataFrame(brows)
    bdf.to_csv(out_dir / "bootstrap.csv", index=False)

    # ---- reliability diagnostics (D/E/F): mean r per run x condition; clean NORM vs abnormal
    norm_only = (Y[:, 0] == 1) & (Y[:, 1:].sum(1) == 0)
    abnormal = Y[:, 0] == 0
    rrows = []
    for k, rc in rmean.items():
        for cname, rr in rc.items():
            c = cond_by_name[cname]
            base = dict(variant=k[0], regime=k[1], seed=k[2], condition=cname, family=c.family,
                        severity=c.severity, mode=c.mode)
            rrows.append({**base, "subset": "all", "n": len(rr), "mean_r": float(rr.mean())})
            if c.family == "clean":
                for sname, msk in (("clean_norm", norm_only), ("clean_abnormal", abnormal)):
                    rrows.append({**base, "subset": sname, "n": int(msk.sum()),
                                  "mean_r": float(rr[msk].mean()) if msk.any() else np.nan})
    rel = pd.DataFrame(rrows, columns=["variant", "regime", "seed", "condition", "family", "severity",
                                       "mode", "subset", "n", "mean_r"])
    rel.to_csv(out_dir / "reliability.csv", index=False)

    # ---- figures
    fam_order = list(ALL_FAMILIES) + ["mixed"]
    perf_fs = (per_run.groupby(["variant", "regime", "seed", "family", "severity"])[["macro_auroc", "macro_f1"]]
               .mean().reset_index())  # mode-averaged per seed
    variants_present = [v for v in "ABCDEFG" if v in set(per_run.variant)]
    for reg in ("aug", "clean"):
        if (perf_fs.regime == reg).any():
            plot_vs_severity(perf_fs, "macro_auroc", fam_order, variants_present, [reg], "macro-AUROC",
                             f"Test macro-AUROC vs severity (regime {reg}; mean ± std over seeds)",
                             fig_dir / f"auroc_vs_snr_{reg}.png")
    if len(rel):
        rel_fs = (rel[rel.subset == "all"].groupby(["variant", "regime", "seed", "family", "severity"])["mean_r"]
                  .mean().reset_index())
        plot_vs_severity(rel_fs, "mean_r", fam_order, [v for v in R_VARIANTS if v in set(rel.variant)],
                         ["aug", "clean"], "mean r", "Mean reliability r vs severity (mean ± std over seeds)",
                         fig_dir / "r_vs_snr.png")

    # ---- decision rule inputs (aug-regime candidates)
    cands = {}
    for cand in [c for c in CANDIDATES if c in aug_models]:
        def ci_of(g, sev):
            b = best_base.get((g, sev))
            if b is None:
                return None
            row = bdf[(bdf.candidate == cand) & (bdf.baseline == b) & (bdf.group == g) & (bdf.severity == sev)].iloc[0]
            return (b, row["diff"], row.ci_low, row.ci_high)
        rc = rel[(rel.variant == cand) & (rel.regime == "aug")]
        r_sev, r_abn, r_norm = None, None, None
        if len(rc):
            corr = rc[(rc.subset == "all") & (rc.family != "clean")]
            per_seed = corr.groupby(["seed", "severity"])["mean_r"].mean().unstack()
            r_sev = {s: float(per_seed[s].mean()) if s in per_seed else np.nan for s in SEV_ORDER}
            r_abn = float(rc[rc.subset == "clean_abnormal"]["mean_r"].mean())
            r_norm = float(rc[rc.subset == "clean_norm"]["mean_r"].mean())
        cands[cand] = {"ci": {g: ci_of(g, "all") for g in binding},
                       "supporting": {f"{g} ({sev})": ci_of(g, sev) for g, sev in SUPPORTING if g in gfam},
                       "r_sev": r_sev, "r_abn": r_abn, "r_norm": r_norm}
    decision = apply_decision_rule(cands, E["r_abnormal_tolerance"], binding)
    write_summary(out_dir / "summary.md", cfg, runs, decision, agg, bdf, rel, len(Y), len(np.unique(pids)),
                  gfam, binding, pool)
    print(f"verdict: {decision['verdict']}  -> {out_dir / 'summary.md'}")
    return decision


def write_summary(path: Path, cfg, runs, decision, agg, bdf, rel, n_test, n_pat,
                  gfam: dict, binding: list[str], pool: list[str]) -> None:
    E = cfg["eval"]
    present = sorted({r["variant"] for r in runs if r["regime"] == "aug"})
    cands = [v for v in CANDIDATES if v in present]
    bases = [v for v in BASELINES if v in present]
    L = ["# RACER Phase 1 — summary", ""]
    if cfg.get("smoke"):
        L += ["> **SMOKE RUN on synthetic data — numbers are meaningless.**", ""]
    L += [f"## Verdict: **{decision['verdict']}**", ""]
    if decision["passing"]:
        L += [f"Candidate(s) passing every binding criterion: {', '.join(decision['passing'])}.", ""]
    elif decision["verdict"] == "NO-GO / PIVOT":
        L += [f"Augmented baselines match RACER-lite ({'/'.join(cands) or 'E/F'}) on at least one binding criterion, or r does not "
              "track severity, or r drops on clean-abnormal ECGs. **The outcome supports a "
              "benchmark-and-analysis paper instead.**", ""]
    if decision["missing"]:
        L += [f"Missing inputs: {', '.join(decision['missing'])}.", ""]
    L += ["## B8 criteria", "", "| candidate | criterion | value | rule | pass | binding |", "|---|---|---|---|---|---|"]
    for r in decision["criteria"]:
        L.append(f"| {r['candidate']} | {r['criterion']} | {r['value']} | {r['rule']} | "
                 f"{'PASS' if r['passed'] else 'FAIL'} | {'yes' if r['binding'] else 'no'} |")
    L += ["", "Operationalization (fixed before seeing results):",
          f"- Candidates: {', '.join(cands) or 'none'} trained in regime aug. Baselines: augmented "
          f"{', '.join(bases) or 'none'}. The *best* baseline for a "
          f"group is the one with the highest seed-averaged-probability macro-AUROC on that group.",
          f"- Binding groups: {', '.join(binding)}, pooled over severities and injection modes (group "
          f"AUROC = unweighted mean over its conditions). 0 dB and unseen-family comparisons are reported as "
          f"supporting (non-binding) evidence.",
          f"- CI: paired bootstrap over test patients ({E['n_bootstrap']} resamples of {n_pat} patient clusters, "
          f"{n_test} records), 95% percentile interval, on macro-AUROC of seed-averaged predicted probabilities.",
          "- r monotonicity: mean r of the candidate over all corrupted test conditions at each severity "
          "(all families incl. mixed, both modes), averaged over seeds; strict mild > moderate > severe.",
          f"- Clean-abnormal = no NORM label; clean-NORM = NORM as the only label; pass if r_abn >= r_norm - "
          f"{E['r_abnormal_tolerance']}.",
          f"- GO iff one candidate ({' or '.join(cands) or 'E or F'}) passes all binding criteria.", ""]

    L += ["## Macro-AUROC (mean ± std over seeds)", "",
          "Group value = unweighted mean over its conditions; drop = clean − corrupted.", ""]
    cols = [(g, s) for g, s in [("clean", "none"), ("real_noise", "all"), ("real_noise", "severe"),
                                ("mixed", "all"), ("mixed", "severe"), ("seen_families", "all"),
                                ("unseen_families", "all"), ("unseen_no_em", "all"), ("all_corrupted", "all")]
            if g == "clean" or g in gfam]
    L += ["| variant | regime | " + " | ".join(f"{g} ({s})" if s != "none" else g for g, s in cols) + " |",
          "|---|---|" + "---|" * len(cols)]
    ai = agg.set_index(["variant", "regime", "group", "severity"])
    for (v, reg) in sorted({(a, b) for a, b in zip(agg.variant, agg.regime)}):
        cells = []
        for g, s in cols:
            k = (v, reg, g, s)
            cells.append(fmt_ms(ai.loc[k, "auroc_mean"], ai.loc[k, "auroc_std"]) if k in ai.index else "n/a")
        L.append(f"| {v} | {reg} | " + " | ".join(cells) + " |")
    L += ["", "Drop in macro-AUROC (clean − corrupted), mean ± std over seeds:", "",
          "| variant | regime | " + " | ".join(f"{g} ({s})" for g, s in cols[1:]) + " |",
          "|---|---|" + "---|" * (len(cols) - 1)]
    for (v, reg) in sorted({(a, b) for a, b in zip(agg.variant, agg.regime)}):
        cells = []
        for g, s in cols[1:]:
            k = (v, reg, g, s)
            cells.append(fmt_ms(ai.loc[k, "drop_auroc_mean"], ai.loc[k, "drop_auroc_std"]) if k in ai.index else "n/a")
        L.append(f"| {v} | {reg} | " + " | ".join(cells) + " |")

    if len(bdf):
        L += ["", "## Paired bootstrap vs best augmented baseline", "",
              "| candidate | group | severity | best baseline | diff | 95% CI |", "|---|---|---|---|---|---|"]
        for _, r in bdf[bdf.is_best_baseline].iterrows():
            L.append(f"| {r.candidate} | {r.group} | {r.severity} | {r.baseline} | {r['diff']:+.4f} | "
                     f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] |")
        L += ["", "All pairwise comparisons: `bootstrap.csv`."]

    if len(rel):
        L += ["", "## Reliability r (mean over seeds)", "",
              "| variant | regime | clean-NORM | clean-abnormal | mild | moderate | severe |", "|---|---|---|---|---|---|---|"]
        for (v, reg), sub in rel.groupby(["variant", "regime"]):
            cn = sub[sub.subset == "clean_norm"].mean_r.mean()
            ca = sub[sub.subset == "clean_abnormal"].mean_r.mean()
            cr = sub[(sub.subset == "all") & (sub.family != "clean")]
            sv = cr.groupby(["seed", "severity"]).mean_r.mean().groupby("severity").mean()
            L.append(f"| {v} | {reg} | {cn:.4f} | {ca:.4f} | " + " | ".join(
                f"{sv.get(s, np.nan):.4f}" for s in SEV_ORDER) + " |")
        L += ["", "Per-family r vs severity: `reliability.csv`, `figures/r_vs_snr.png`."]

    C = cfg["corruptions"]
    L += ["", "## Notes", "",
          f"- Data: PTB-XL, lead {cfg['data']['lead']} at {cfg['data']['fs']} Hz, test fold(s) "
          f"{cfg['data']['test_folds']}. Training corruption families: {', '.join(C['train_families'])}; "
          f"unseen at test: {', '.join(C['unseen_families'])}."]
    if "nstdb_em" in C["unseen_families"]:
        L.append("- NSTDB `em` (an unseen test family) reportedly contains residual ECG (a persistent regular rhythm) "
                 "in both channels (bioRxiv 10.1101/2022.10.18.512701). Unseen-family results are therefore also "
                 "reported with em excluded (`unseen_no_em`).")
    if "powerline" in C["unseen_families"] + C["train_families"]:
        L.append("- Powerline (50 Hz) is largely removed by the 0.5–40 Hz band-pass applied after corruption; it is "
                 "a near-null sanity-check condition.")
    if not any(f.startswith("nstdb_") for f in C["unseen_families"] + C["train_families"]):
        L.append("- No real (NSTDB) noise in this phase: all corruptions are synthetic.")
    L += [f"- 'mixed' test condition: per record, 2 distinct families drawn from {{{', '.join(pool)}}} "
          "(all configured additive families), summed at unit power and scaled to the target SNR; "
          "× 3 severities × 2 modes.",
          "", "## Runs", "", "| run | params | device | amp | best epoch | val AUROC | git |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        L.append(f"| {r['dir'].name} | {r['n_params']:,} | {r['device']} | {r['amp_dtype']} | {r['best_epoch']} | "
                 f"{r['best_val_macro_auroc']:.4f} | {r['git_hash'][:10]}{'*' if r.get('git_dirty') else ''} |")
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--runs", default=None, help="runs directory (default: paths.runs_dir from config)")
    ap.add_argument("--smoke", action="store_true", help="apply the config's smoke overrides")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.smoke)
    runs_dir = Path(a.runs) if a.runs else REPO / cfg["paths"]["runs_dir"]
    evaluate(cfg, runs_dir.resolve())


if __name__ == "__main__":
    main()
