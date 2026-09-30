"""Gate for the multi-seed stage. Prints the model(s) to run seeds 1-4 for, or nothing.
Rule (agreed 2026-09-29): proceed iff best seed-0 TEST macro-AUROC >= 0.92; choose model by VAL
macro-AUROC; run both if their val macro-AUROC differ by < 0.003."""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
r = {m: json.load(open(os.path.join(R, f"{m}_s0.json"))) for m in ("M1", "M2")}
if max(x["test_macro_auroc"] for x in r.values()) >= 0.92:
    v = {m: x["val_macro_auroc"] for m, x in r.items()}
    print("M1 M2" if abs(v["M1"] - v["M2"]) < 0.003 else max(v, key=v.get))
