"""CinC 2017 report (RULES2.md section 3): single-lead clean fold-9 scores, noisy-recording detection with r vs
heuristics, and the frozen-representation probe. Reads results/track2/cinc2017/*. Writes results/track2/cinc2017.{md,json}."""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import macro_auroc
from sl_infer import GATED, SL
from t2lib import D, auroc, f4, mean_std

C = os.path.join(D, "cinc2017")
z = np.load(os.path.join(C, "data.npz"), allow_pickle=True)
H = np.load(os.path.join(C, "heuristics.npz"))
LAB = {"REFERENCE.csv": z["label"], "REFERENCE-v3.csv": z["label_v3"]}
N = len(z["ids"])
load = lambda norm, t: np.load(os.path.join(C, norm, f"{t}.npz"))
out = {}

# ---------------------------------------------------------------- A. clean fold-9 scores of the single-lead models
RUNS = os.path.join(D, "runs_sl")
clean = {}
for k, tags in SL.items():
    per = [json.load(open(os.path.join(RUNS, f"{t}.json")))["val_macro_auroc"] for t in tags]
    pv = [np.load(os.path.join(RUNS, "probs", f"{t}.npz")) for t in tags]
    clean[k] = {"per_seed": per, "avg3": macro_auroc(pv[0]["y_val"], np.mean([p["val"] for p in pv], 0)),
                "best_epoch": [json.load(open(os.path.join(RUNS, f"{t}.json")))["best_epoch"] for t in tags]}
out["clean_fold9"] = clean


# ---------------------------------------------------------------- C. noisy detection
def boot_idx(n=1000, seed=0, y=None):
    rng = np.random.default_rng(seed)
    res = []
    while len(res) < n:
        i = rng.integers(0, N, N)
        if 0 < y[i].sum() < len(i):
            res.append(i)
    return res


def r_avg(norm, k, stat):
    return np.mean([load(norm, t)[f"r_{stat}"] for t in SL[k]], 0)


det = {}
for lab_name, lab in LAB.items():
    y = (lab == "~").astype(float)
    B = boot_idx(y=y)
    scores = {"H1": H["H1"], "H2": H["H2"]}
    for norm in ["pooled", "record"]:
        for stat in ["mean", "min"]:
            for k in GATED:
                scores[f"{k}|{norm}|{stat}"] = -r_avg(norm, k, stat)        # low r = noisy
    full = {s: auroc(y, v) for s, v in scores.items()}
    bs = {s: np.array([auroc(y[i], v[i]) for i in B]) for s, v in scores.items()}
    best_h = max(["H1", "H2"], key=lambda h: full[h])
    res = {"n_noisy": int(y.sum()), "best_heuristic": best_h, "auc": {}}
    for s in scores:
        d = bs[s] - bs[best_h]
        res["auc"][s] = {"auc": full[s], "lo": float(np.quantile(bs[s], .025)), "hi": float(np.quantile(bs[s], .975)),
                         "diff_vs_best_h": full[s] - full[best_h], "diff_lo": float(np.quantile(d, .025)),
                         "diff_hi": float(np.quantile(d, .975))}
    for k in GATED:
        a = res["auc"][f"{k}|pooled|mean"]
        res.setdefault("claim", {})[k] = bool(a["lo"] > 0.5 and a["diff_lo"] > 0)
    res["per_seed"] = {k: [auroc(y, -load("pooled", t)["r_mean"]) for t in SL[k]] for k in GATED}
    res["mean_r_by_class"] = {k: {c: float(r_avg("pooled", k, "mean")[lab == c].mean()) for c in "NAO~"} for k in GATED}
    det[lab_name] = res
out["noisy_detection"] = det

# ---------------------------------------------------------------- D. probe
lab = LAB["REFERENCE.csv"]
classes = np.array(["N", "A", "O", "~"])
yc = np.searchsorted(classes[np.argsort(classes)], lab)
order = classes[np.argsort(classes)]                     # sorted label order used by yc
probe = {}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
splits = list(cv.split(np.zeros(N), yc))
for k in ["B0-clean", "B0-aug", "F"]:
    probe[k] = []
    for t in SL[k]:
        X = load("pooled", t)["feat"]
        f1s, aucs = [], []
        for tr, te in splits:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
            clf.fit(X[tr], yc[tr])
            pred = clf.predict(X[te])
            per = f1_score(yc[te], pred, labels=[list(order).index(c) for c in "NAO"], average=None)
            f1s.append(float(np.mean(per)))
            aucs.append(float(roc_auc_score(yc[te] == list(order).index("A"), clf.predict_proba(X[te])[:, list(order).index("A")])))
        probe[k].append({"tag": t, "f1_NAO": f1s, "af_auroc": aucs, "n_features": X.shape[1]})
out["probe"] = probe
json.dump(out, open(os.path.join(D, "cinc2017.json"), "w"), indent=1)

# ---------------------------------------------------------------- report
lens = H["length"]
L = ["# CinC 2017 - single-lead transfer of the Track 2 models (no retraining, no tuning)", "",
     "Rules: RULES2.md section 3 (committed before any single-lead run; label-file amendment before any CinC run).", "",
     "**Notes.** CinC 2017 has no patient IDs, so all resampling and CV folds are record-level (one person may "
     "contribute several recordings). The hidden challenge test set is not available, so no score here is comparable "
     "to the leaderboard. Nothing was retrained or tuned on CinC 2017; the probe's logistic regression uses fixed "
     "defaults (C = 1). The PTB-XL models see lead I; CinC 2017 is a handheld lead-I-like recording (AliveCor).", "",
     "## A. Single-lead PTB-XL models (lead I), clean fold-9 macro-AUROC", "",
     "| model | seed 0 | seed 1 | seed 2 | mean +- std | 3-seed avg | best epochs |", "|---|---|---|---|---|---|---|"]
for k, v in clean.items():
    L.append(f"| {k} | " + " | ".join(f4(x) for x in v["per_seed"]) + f" | {mean_std(v['per_seed'])} | {f4(v['avg3'])} | "
             + ", ".join(map(str, v["best_epoch"])) + " |")
L += ["", "For reference, 12-lead B0-clean (Track 2): 0.9356 +- 0.0008 per seed, 0.9389 3-seed avg.", "",
      "## B. Data", "",
      f"- {N} recordings, 300 Hz -> 100 Hz (polyphase 1/3), band-pass 0.5-40 Hz. Length {lens.min() / 100:.1f}-{lens.max() / 100:.1f} s "
      f"(median {np.median(lens) / 100:.1f} s); {int(np.sum(lens < 1000))} recordings < 10 s were reflect-padded.",
      f"- Windows: 10 s, stride 5 s, plus a final window ending at the last sample; {int(H['nwin'].sum())} windows "
      f"({int(H['nwin'].min())}-{int(H['nwin'].max())} per recording).",
      f"- Main standardization: CinC 2017 pooled mean/std over all samples (mean {float(H['mu']):.4f} mV, sd {float(H['sd']):.4f} mV). "
      "Sensitivity: per-record z-score.",
      "- Labels: " + "; ".join(f"{n}: " + ", ".join(f"{c} {int((l == c).sum())}" for c in "NAO~") for n, l in LAB.items()) + ".", "",
      "## C. Noisy-recording detection (\"~\" vs rest)", "",
      "Score = -r (low r = noisy), r = 3-seed average of the per-recording r (mean or min over windows). "
      "AUROC with recording-bootstrap 95% CI (1000 resamples, seed 0); paired bootstrap of AUC(r) - AUC(best heuristic) on the same resamples. "
      "H1 = power(20-40 Hz) / power(0.5-20 Hz); H2 = max |z-scored amplitude|; both high = noisy.", ""]
for lab_name, res in det.items():
    L += [f"### Labels: {lab_name} ({res['n_noisy']} noisy){' - main' if lab_name == 'REFERENCE.csv' else ' - sensitivity'}", "",
          f"Best heuristic: **{res['best_heuristic']}**. H1 AUROC {f4(res['auc']['H1']['auc'])} [{res['auc']['H1']['lo']:.4f}, {res['auc']['H1']['hi']:.4f}]; "
          f"H2 {f4(res['auc']['H2']['auc'])} [{res['auc']['H2']['lo']:.4f}, {res['auc']['H2']['hi']:.4f}].", "",
          "| model | norm | r stat | AUROC [95% CI] | AUC(r) - AUC(best heuristic) [95% CI] |", "|---|---|---|---|---|"]
    for norm in ["pooled", "record"]:
        for stat in ["mean", "min"]:
            for k in GATED:
                a = res["auc"][f"{k}|{norm}|{stat}"]
                main = norm == "pooled" and stat == "mean"
                L.append(f"| {'**' + k + '**' if main else k} | {norm} | {stat} | {f4(a['auc'])} [{a['lo']:.4f}, {a['hi']:.4f}] | "
                         f"{a['diff_vs_best_h']:+.4f} [{a['diff_lo']:+.4f}, {a['diff_hi']:+.4f}] |")
    L += ["", "Per seed (pooled, mean r): " + "; ".join(f"{k} " + " / ".join(f4(x) for x in v) for k, v in res["per_seed"].items()) + ".", "",
          "Mean r by class (pooled, 3-seed): " + "; ".join(f"{k}: " + ", ".join(f"{c} {v[c]:.3f}" for c in "NAO~") for k, v in res["mean_r_by_class"].items()) + ".", "",
          "Claim \"r separates Noisy\" (main analysis: pooled, mean r; needs AUC CI lower bound > 0.5 AND paired CI vs best heuristic above 0): "
          + "; ".join(f"{k} **{'yes' if v else 'no'}**" for k, v in res["claim"].items()) + ".", ""]
L += ["## D. Frozen-representation probe (4-class, labels REFERENCE.csv)", "",
      "Features averaged over crops and windows: B0 = 256-d concat pool, F = 64-d fused feature. StandardScaler + "
      "LogisticRegression (C = 1, max_iter 5000); stratified 5-fold CV by recording, repeated 3 times (random_state 0). "
      "F1 = mean(F1_N, F1_A, F1_O) (challenge definition); AF AUROC = A vs rest. Mean +- std over the 15 test folds.", "",
      "| model | features | seed 0 F1 | seed 1 F1 | seed 2 F1 | mean F1 over seeds | seed 0 AF AUROC | seed 1 | seed 2 | mean AF AUROC |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for k, runs in probe.items():
    L.append(f"| {k} | {runs[0]['n_features']} | " + " | ".join(mean_std(r["f1_NAO"]) for r in runs) +
             f" | {f4(np.mean([np.mean(r['f1_NAO']) for r in runs]))} | " + " | ".join(mean_std(r["af_auroc"]) for r in runs) +
             f" | {f4(np.mean([np.mean(r['af_auroc']) for r in runs]))} |")
L += ["", "The feature sizes differ (256 vs 64), so the probe compares representations, not equal-capacity classifiers."]
open(os.path.join(D, "cinc2017.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
