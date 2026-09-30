# Fold-10 evaluation log

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
| 9 | 2026-10-01 00:48 | calibration: resnet1d_wang_crop_dsnorm | resnet1d_wang_crop_dsnorm | Track 2 step 2 calibration (no tuning on fold 10) |
| 10 | 2026-10-01 00:48 | calibration: resnet1d_wang_crop_dsnorm_s1 | resnet1d_wang_crop_dsnorm_s1 | Track 2 step 2 calibration (no tuning on fold 10) |
| 11 | 2026-10-01 00:48 | calibration: resnet1d_wang_crop_dsnorm_s2 | resnet1d_wang_crop_dsnorm_s2 | Track 2 step 2 calibration (no tuning on fold 10) |
| 12 | 2026-10-01 00:48 | calibration: 3-seed average | resnet1d_wang_crop_dsnorm, resnet1d_wang_crop_dsnorm_s1, resnet1d_wang_crop_dsnorm_s2 | Track 2 step 2 calibration, mean of the 3 probabilities above |
