"""Print M1 vs M2 (seed 0) vs published PTB-XL superdiagnostic references (Strodthoff et al. 2021)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = {"xresnet1d101": .928, "resnet1d_wang": .930, "inception1d": .921, "ensemble": .934, "naive": .500}

rows = []
for m in ("M1", "M2"):
    p = os.path.join(HERE, "results", f"{m}_s0.json")
    if os.path.exists(p):
        r = json.load(open(p))
        rows.append((f"{m} (seed 0, ours)", r["test_macro_auroc"], r["val_macro_auroc"], r["n_params"], r["best_epoch"]))
print(f"{'model':28s} {'test mAUROC':>11s} {'val mAUROC':>10s} {'params':>9s} {'best_ep':>7s}")
for n, t, v, p, b in rows:
    print(f"{n:28s} {t:11.4f} {v:10.4f} {p:9d} {b:7d}")
for n, t in REFS.items():
    print(f"{n + ' (published)':28s} {t:11.3f} {'-':>10s} {'-':>9s} {'-':>7s}")
