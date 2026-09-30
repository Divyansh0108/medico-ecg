"""Difficulty check (fold 9, clean-trained resnet1d_wang seeds 0-2, no retraining) and the severity rule
of RULES.md section C. Writes results/track2/difficulty.json and difficulty.md."""
import json
import os

import numpy as np

from corruptions import ALL_CONDITION_FAMILIES, MODES, SNRS, cond_name
from metrics import macro_auroc

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "track2")
TAGS = ["resnet1d_wang_crop_dsnorm", "resnet1d_wang_crop_dsnorm_s1", "resnet1d_wang_crop_dsnorm_s2"]
BAR = 0.03

Z = {t: np.load(os.path.join(D, "grid", "difficulty", f"{t}.npz")) for t in TAGS}
names = list(Z[TAGS[0]]["conds"])
auc = {t: {c: macro_auroc(Z[t]["y"], Z[t]["probs"][k]) for k, c in enumerate(names)} for t in TAGS}
drop = {t: {c: auc[t]["clean"] - auc[t][c] for c in names} for t in TAGS}
clean = np.array([auc[t]["clean"] for t in TAGS])

L = ["## Difficulty check (fold 9, B0-clean = resnet1d_wang_crop_dsnorm seeds 0-2, no retraining)", "",
     f"Clean macro-AUROC per seed: {' '.join(f'{v:.4f}' for v in clean)} (mean {clean.mean():.4f}).",
     "Cells: mean over the 3 seeds of macro-AUROC (drop vs clean).", "",
     "| family | mode | " + " | ".join(f"{s:+.0f} dB" for s in SNRS) + " |", "|---|---|" + "---|" * len(SNRS)]
for f in ALL_CONDITION_FAMILIES:
    for m in MODES:
        cells = []
        for s in SNRS:
            c = cond_name(f, s, m)
            cells.append(f"{np.mean([auc[t][c] for t in TAGS]):.4f} ({np.mean([drop[t][c] for t in TAGS]):+.3f})")
        L.append(f"| {f} | {m} | " + " | ".join(cells) + " |")

mixed_drop = {s: float(np.mean([drop[t][cond_name("mixed", s, m)] for t in TAGS for m in MODES])) for s in SNRS}
chosen = next((s for s in SNRS if mixed_drop[s] >= BAR), None)
L += ["", "Severity rule: mean drop on mixed (3 seeds x 2 modes): " +
      ", ".join(f"{s:+.0f} dB {v:.4f}" for s, v in mixed_drop.items()) + f"; bar {BAR}."]
if chosen is None:
    L.append("**Even -6 dB gives a drop < 0.03: the backbone is robust to this benchmark. STOP.**")
    sev = []
else:
    sev = [s for s in SNRS if s <= chosen]
    L.append(f"**Mildest SNR meeting the bar: {chosen:+.0f} dB. Chosen severity set: {', '.join(f'{s:+.0f}' for s in sev)} dB.**")
json.dump({"auroc": auc, "drop": drop, "mixed_mean_drop": mixed_drop, "chosen_snr": chosen, "severity_set": sev},
          open(os.path.join(D, "difficulty.json"), "w"), indent=1)
open(os.path.join(D, "difficulty.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
