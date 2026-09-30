# PTB-XL Track 1 - seeds, SWA/EMA, label smoothing, crop 500, ensemble

Recipe: 2.5 s crops, sliding-window mean (250/125), dataset norm fitted on train, band-pass on.
Selection on fold 9 only; rule and decision in SELECTION2.md.

## 1. Seeds 0-4 (fold 9)

| model | per seed (0-4) | mean +- std | 5-seed probability average |
|---|---|---|---|
| resnet1d_wang | 0.9365 0.9356 0.9349 0.9368 0.9349 | 0.9357 +- 0.0009 | 0.9394 |
| inception1d | 0.9346 0.9339 0.9327 0.9345 0.9330 | 0.9337 +- 0.0008 | 0.9389 |
| xresnet1d50 | 0.9333 0.9336 0.9318 0.9316 0.9321 | 0.9325 +- 0.0009 | 0.9375 |

## 2. Seed-0 variants of resnet1d_wang (fold 9), noise floor = seed std 0.0009

| variant | fold-9 mAUROC | vs base | same run, best epoch |
|---|---|---|---|
| base (best epoch, early stopping) | 0.9365 | +0.0000 | (ep 28) |
| SWA, epochs 41-50 | 0.9329 | -0.0036 | 0.9365 (ep 28) |
| EMA 0.999, epochs 41-50 | 0.9327 | -0.0038 | 0.9365 (ep 28) |
| label smoothing 0.05 | 0.9382 | +0.0018 | 0.9382 (ep 28) |
| crop 500, stride 250 | 0.9366 | +0.0001 | 0.9366 (ep 28) |

## 3. Pre-specified ensemble (fold 9)

Mean of probabilities, seeds 0-4 of resnet1d_wang, inception1d, xresnet1d50 (15 models): **0.9403**

## 4. Fold 10 (each entry evaluated once)

| entry | members | fold 9 | fold 10 | 95% CI (patient bootstrap) |
|---|---|---|---|---|
| FINAL2_single | 1 | 0.9382 | **0.9335** | 0.9259 - 0.9401 |
| FINAL2_ensemble | 15 | 0.9403 | **0.9363** | 0.9291 - 0.9428 |
| FINAL_single | 1 | 0.9365 | **0.9332** | 0.9255 - 0.9399 |
| BASE_xresnet1d101 | 1 | 0.9311 | **0.9309** | 0.9233 - 0.9378 |

Paired patient bootstrap (1000 resamples, same patients for both), fold-10 macro-AUROC difference:

| ours | baseline | ours - baseline | 95% CI | P(diff <= 0) | baseline published |
|---|---|---|---|---|---|
| FINAL2_single | resnet1d_wang (reproduced) | +0.0002 | -0.0015 to +0.0018 | 0.398 | 0.930 |
| FINAL2_single | xresnet1d101 (reproduced) | +0.0026 | -0.0002 to +0.0054 | 0.034 | 0.928 |
| FINAL2_ensemble | resnet1d_wang (reproduced) | +0.0031 | +0.0012 to +0.0051 | 0.000 | 0.930 |
| FINAL2_ensemble | xresnet1d101 (reproduced) | +0.0054 | +0.0031 to +0.0076 | 0.000 | 0.928 |

Fold-10 scores of the ensemble members (printed by final_eval.py, not used for any choice):

| model | per seed (0-4) | mean +- std |
|---|---|---|
| resnet1d_wang | 0.9332 0.9307 0.9311 0.9308 0.9315 | 0.9315 +- 0.0010 |
| inception1d | 0.9298 0.9297 0.9278 0.9314 0.9289 | 0.9295 +- 0.0013 |
| xresnet1d50 | 0.9300 0.9273 0.9267 0.9272 0.9289 | 0.9280 +- 0.0014 |

Published (Strodthoff et al. 2021): resnet1d_wang .930, xresnet1d101 .928, inception1d .921, ensemble .934.
