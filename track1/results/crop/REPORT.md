# PTB-XL Track 1 - crops, backbones, ablations (seed 0)

Selection on fold 9 only. Fold 10 is evaluated only via final_eval.py (table 3).

## 1. All runs (fold 9)

| run | model | fold-9 mAUROC (mean agg) | fold-9 (max agg) | params | best ep / run | train time |
|---|---|---|---|---|---|---|
| M1_crop | M1 | 0.9144 | 0.9123 | 1,046,557 | 33 / 43 | 8.5 min |
| M1_crop_datasetnorm | M1 | 0.9327 | 0.9310 | 1,046,557 | 38 / 48 | 9.5 min |
| M1_crop_nobandpass | M1 | 0.9144 | 0.9121 | 1,046,557 | 33 / 43 | 8.7 min |
| M1_crop_nomixup | M1 | 0.9124 | 0.9103 | 1,046,557 | 13 / 23 | 4.6 min |
| M1_crop_onecycle | M1 | 0.8971 | 0.8931 | 1,046,557 | 2 / 12 | 2.5 min |
| inception1d_crop | inception1d | 0.9170 | 0.9158 | 507,653 | 26 / 36 | 12.3 min |
| inception1d_crop_dsnorm | inception1d | 0.9346 | 0.9334 | 507,653 | 26 / 36 | 12.1 min |
| resnet1d_wang_crop | resnet1d_wang | 0.9178 | 0.9160 | 473,349 | 21 / 31 | 3.4 min |
| resnet1d_wang_crop_dsnorm | resnet1d_wang | 0.9365 | 0.9337 | 473,349 | 28 / 38 | 4.3 min |
| xresnet1d101_crop | xresnet1d101 | 0.9149 | 0.9142 | 1,872,261 | 25 / 35 | 21.8 min |
| xresnet1d101_crop_dsnorm | xresnet1d101 | 0.9311 | 0.9294 | 1,872,261 | 26 / 36 | 22.8 min |
| xresnet1d50_crop | xresnet1d50 | 0.9174 | 0.9165 | 953,989 | 25 / 35 | 12.4 min |
| xresnet1d50_crop_dsnorm | xresnet1d50 | 0.9333 | 0.9317 | 953,989 | 28 / 38 | 13.5 min |

## 2. Fold-9 ensembles of backbone runs (mean of probabilities)

| members | fold-9 mAUROC |
|---|---|
| resnet1d_wang_crop + M1_crop_datasetnorm + resnet1d_wang_crop_dsnorm + xresnet1d50_crop_dsnorm + inception1d_crop_dsnorm | 0.9395 |
| inception1d_crop + M1_crop_datasetnorm + resnet1d_wang_crop_dsnorm + xresnet1d50_crop_dsnorm + inception1d_crop_dsnorm | 0.9394 |
| xresnet1d50_crop + M1_crop_datasetnorm + resnet1d_wang_crop_dsnorm + xresnet1d50_crop_dsnorm + inception1d_crop_dsnorm | 0.9394 |
| M1_crop_datasetnorm + resnet1d_wang_crop_dsnorm + xresnet1d50_crop_dsnorm + inception1d_crop_dsnorm | 0.9393 |
| resnet1d_wang_crop + resnet1d_wang_crop_dsnorm + xresnet1d50_crop_dsnorm + inception1d_crop_dsnorm | 0.9393 |
| resnet1d_wang_crop_dsnorm + xresnet1d50_crop_dsnorm + inception1d_crop_dsnorm | 0.9392 |
| xresnet1d50_crop + resnet1d_wang_crop_dsnorm + xresnet1d50_crop_dsnorm + inception1d_crop_dsnorm | 0.9392 |
| resnet1d_wang_crop + M1_crop_datasetnorm + resnet1d_wang_crop_dsnorm + xresnet1d50_crop_dsnorm + xresnet1d101_crop_dsnorm + inception1d_crop_dsnorm | 0.9391 |

## 3. Fold 10 (evaluated once per entry) vs published

| model | fold-9 | fold-10 mAUROC | 95% CI (patient bootstrap) |
|---|---|---|---|
| **FINAL_ensemble** (M1_crop_datasetnorm + resnet1d_wang_crop_dsnorm + xresnet1d50_crop_dsnorm + xresnet1d101_crop_dsnorm + inception1d_crop_dsnorm) | 0.9388 | **0.9364** | 0.9289 - 0.9430 |
| **FINAL_single** (resnet1d_wang_crop_dsnorm) | 0.9365 | **0.9332** | 0.9255 - 0.9399 |
| **M1_crop_fold10** (M1_crop) | 0.9144 | **0.9129** | 0.9042 - 0.9202 |
| M1 full-length, no crops (previous run) | 0.8994 | 0.8967 | - |
| xresnet1d101 (Strodthoff et al. 2021, published) | - | 0.928 | - |
| resnet1d_wang (Strodthoff et al. 2021, published) | - | 0.930 | - |
| inception1d (Strodthoff et al. 2021, published) | - | 0.921 | - |
| ensemble (Strodthoff et al. 2021, published) | - | 0.934 | - |
