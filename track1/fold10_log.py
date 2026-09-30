"""Every fold-10 evaluation goes through log_fold10(), which appends a numbered row to
results/track2/FOLD10_LOG.md. One row = one entry (a single model or an ensemble) scored on fold 10."""
from __future__ import annotations

import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "results", "track2", "FOLD10_LOG.md")
HEADER = """# Fold-10 evaluation log

One row = one entry (single model or ensemble) scored on fold 10. Rows 1-8 are from before Track 2,
reconstructed from the result files. Every later row is written by fold10_log.log_fold10() when the
scoring happens.

| # | time | entry | models | purpose |
|---|---|---|---|---|
| 1 | 2026-09-29 | M1_s0 (full length) | M1_s0 | round 1, train.py --eval-test |
| 2 | 2026-09-29 | M2_s0 (full length) | M2_s0 | round 1, train.py --eval-test |
| 3 | 2026-09-30 | M1_crop_fold10 | M1_crop | round 2, asked for |
| 4 | 2026-09-30 | FINAL_single | resnet1d_wang_crop_dsnorm | round 2 final single |
| 5 | 2026-09-30 | FINAL_ensemble | 5 dataset-norm models | round 2 final ensemble |
| 6 | 2026-09-30 | FINAL2_single | resnet1d_wang_crop_dsnorm_ls05 | round 3 final single |
| 7 | 2026-09-30 | BASE_xresnet1d101 | xresnet1d101_crop_dsnorm | round 3 baseline for paired bootstrap |
| 8 | 2026-09-30 | FINAL2_ensemble | 15 models (seeds 0-4 x 3 backbones) | round 3 final ensemble |
"""


def log_fold10(entry: str, models: list[str], purpose: str) -> int:
    if not os.path.exists(LOG):
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        open(LOG, "w").write(HEADER)
    n = max(int(m) for m in re.findall(r"^\| (\d+) \|", open(LOG).read(), re.M)) + 1
    with open(LOG, "a") as f:
        f.write(f"| {n} | {time.strftime('%Y-%m-%d %H:%M')} | {entry} | {', '.join(models)} | {purpose} |\n")
    print(f"[fold10 #{n}] {entry}: {purpose}", flush=True)
    return n
