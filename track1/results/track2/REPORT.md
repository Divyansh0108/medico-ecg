# PTB-XL Track 2 - calibration, corruption benchmark, reliability-gated variants

Rules: RULES.md (committed before any Track 2 run; severity set and amendment in section E).
Fold-10 evaluations: FOLD10_LOG.md. Everything below the calibration section is fold 9 only.

## 1. Calibration (resnet1d_wang_crop_dsnorm seeds 0-2, fitted on fold 9, scored on fold 10)

| entry | fold-10 mAUROC (raw = calibrated) | 95% CI | ECE before -> after (mean of 5) | macro-F1 | precision | recall | bal. acc. |
|---|---|---|---|---|---|---|---|
| resnet1d_wang_crop_dsnorm | 0.9332 | 0.9255-0.9399 | 0.1174 -> 0.0190 | 0.7591 | 0.7077 | 0.8244 | 0.8533 |
| resnet1d_wang_crop_dsnorm_s1 | 0.9307 | 0.9228-0.9378 | 0.1217 -> 0.0191 | 0.7607 | 0.7347 | 0.7915 | 0.8464 |
| resnet1d_wang_crop_dsnorm_s2 | 0.9311 | 0.9235-0.9380 | 0.1194 -> 0.0194 | 0.7514 | 0.7170 | 0.7930 | 0.8465 |
| ensemble_s0-2 | 0.9347 | 0.9271-0.9414 | 0.1203 -> 0.0186 | 0.7639 | 0.7405 | 0.7898 | 0.8490 |

Per-class ECE on fold 10 (15 bins), before -> after Platt:

| entry | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| resnet1d_wang_crop_dsnorm | 0.067 -> 0.025 | 0.099 -> 0.017 | 0.114 -> 0.018 | 0.095 -> 0.018 | 0.212 -> 0.017 |
| resnet1d_wang_crop_dsnorm_s1 | 0.056 -> 0.018 | 0.101 -> 0.022 | 0.109 -> 0.012 | 0.112 -> 0.025 | 0.230 -> 0.018 |
| resnet1d_wang_crop_dsnorm_s2 | 0.067 -> 0.017 | 0.107 -> 0.023 | 0.095 -> 0.015 | 0.119 -> 0.023 | 0.208 -> 0.018 |
| ensemble_s0-2 | 0.064 -> 0.028 | 0.105 -> 0.020 | 0.106 -> 0.011 | 0.110 -> 0.020 | 0.217 -> 0.013 |

Thresholds (max F1 on fold 9, calibrated probabilities), applied unchanged to fold 10:

| entry | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| resnet1d_wang_crop_dsnorm | 0.259 | 0.231 | 0.355 | 0.387 | 0.255 |
| resnet1d_wang_crop_dsnorm_s1 | 0.355 | 0.371 | 0.331 | 0.441 | 0.304 |
| resnet1d_wang_crop_dsnorm_s2 | 0.391 | 0.434 | 0.385 | 0.371 | 0.256 |
| ensemble_s0-2 | 0.427 | 0.386 | 0.432 | 0.326 | 0.305 |

## 2. Difficulty check (fold 9, B0-clean = resnet1d_wang_crop_dsnorm seeds 0-2, no retraining)

Clean macro-AUROC per seed: 0.9365 0.9356 0.9349 (mean 0.9356).
Cells: mean over the 3 seeds of macro-AUROC (drop vs clean).

| family | mode | +15 dB | +6 dB | +0 dB | -6 dB |
|---|---|---|---|---|---|
| baseline_wander | whole | 0.9354 (+0.000) | 0.9285 (+0.007) | 0.9075 (+0.028) | 0.8371 (+0.099) |
| baseline_wander | burst | 0.9355 (+0.000) | 0.9349 (+0.001) | 0.9323 (+0.003) | 0.9289 (+0.007) |
| emg | whole | 0.9357 (-0.000) | 0.9306 (+0.005) | 0.9098 (+0.026) | 0.8359 (+0.100) |
| emg | burst | 0.9358 (-0.000) | 0.9352 (+0.000) | 0.9329 (+0.003) | 0.9301 (+0.006) |
| motion_burst | whole | 0.9352 (+0.000) | 0.9282 (+0.007) | 0.9126 (+0.023) | 0.8640 (+0.072) |
| motion_burst | burst | 0.9355 (+0.000) | 0.9321 (+0.004) | 0.9315 (+0.004) | 0.9285 (+0.007) |
| dropout | whole | 0.9340 (+0.002) | 0.9258 (+0.010) | 0.8949 (+0.041) | 0.8119 (+0.124) |
| dropout | burst | 0.9355 (+0.000) | 0.9342 (+0.001) | 0.9323 (+0.003) | 0.9308 (+0.005) |
| powerline | whole | 0.9358 (-0.000) | 0.9331 (+0.003) | 0.9159 (+0.020) | 0.8446 (+0.091) |
| powerline | burst | 0.9357 (-0.000) | 0.9355 (+0.000) | 0.9337 (+0.002) | 0.9301 (+0.006) |
| mixed | whole | 0.9352 (+0.000) | 0.9256 (+0.010) | 0.8949 (+0.041) | 0.7860 (+0.150) |
| mixed | burst | 0.9355 (+0.000) | 0.9345 (+0.001) | 0.9308 (+0.005) | 0.9281 (+0.008) |

Severity rule: mean drop on mixed (3 seeds x 2 modes): +15 dB 0.0003, +6 dB 0.0056, +0 dB 0.0228, -6 dB 0.0786; bar 0.03.
**Mildest SNR meeting the bar: -6 dB. Chosen severity set: -6 dB.**

## 3. Group scores (fold 9, severity set [-6.0] dB, both modes)

3-seed probability average; drop vs the model's own clean score in brackets.

| model | clean | seen_families | unseen_families | mixed | all_corrupted |
|---|---|---|---|---|---|
| B0-clean | 0.9389 | 0.8947 (+0.044) | 0.8985 (+0.040) | 0.8661 (+0.073) | 0.8919 (+0.047) |
| B0-aug | 0.9389 | 0.9241 (+0.015) | 0.9161 (+0.023) | 0.9094 (+0.030) | 0.9177 (+0.021) |
| C | 0.9371 | 0.9201 (+0.017) | 0.9096 (+0.028) | 0.9059 (+0.031) | 0.9125 (+0.025) |
| D | 0.9380 | 0.9231 (+0.015) | 0.9132 (+0.025) | 0.9080 (+0.030) | 0.9156 (+0.022) |
| E | 0.9381 | 0.9231 (+0.015) | 0.9148 (+0.023) | 0.9097 (+0.028) | 0.9167 (+0.021) |
| F | 0.9380 | 0.9257 (+0.012) | 0.9173 (+0.021) | 0.9115 (+0.027) | 0.9191 (+0.019) |
| E-clean | 0.9376 | 0.8604 (+0.077) | 0.8789 (+0.059) | 0.8438 (+0.094) | 0.8669 (+0.071) |

Per seed (seeds 0-2), mean +- std:

| model | clean | seen_families | unseen_families | mixed | all_corrupted |
|---|---|---|---|---|---|
| B0-clean | 0.9356 +- 0.0008 | 0.8830 +- 0.0030 | 0.8850 +- 0.0036 | 0.8570 +- 0.0034 | 0.8797 +- 0.0031 |
| B0-aug | 0.9361 +- 0.0016 | 0.9196 +- 0.0025 | 0.9099 +- 0.0033 | 0.9042 +- 0.0050 | 0.9122 +- 0.0033 |
| C | 0.9339 +- 0.0013 | 0.9134 +- 0.0052 | 0.8995 +- 0.0061 | 0.8969 +- 0.0058 | 0.9037 +- 0.0057 |
| D | 0.9346 +- 0.0011 | 0.9170 +- 0.0012 | 0.9038 +- 0.0024 | 0.9000 +- 0.0010 | 0.9076 +- 0.0014 |
| E | 0.9347 +- 0.0014 | 0.9160 +- 0.0025 | 0.9044 +- 0.0027 | 0.9000 +- 0.0030 | 0.9075 +- 0.0018 |
| F | 0.9348 +- 0.0017 | 0.9207 +- 0.0025 | 0.9085 +- 0.0023 | 0.9041 +- 0.0008 | 0.9118 +- 0.0011 |
| E-clean | 0.9340 +- 0.0013 | 0.8437 +- 0.0168 | 0.8557 +- 0.0096 | 0.8286 +- 0.0147 | 0.8471 +- 0.0126 |

Per condition at -6 dB (3-seed average):

| model | baseline_wander whole | baseline_wander burst | emg whole | emg burst | motion_burst whole | motion_burst burst | dropout whole | dropout burst | powerline whole | powerline burst | mixed whole | mixed burst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0-clean | 0.8579 | 0.9335 | 0.8539 | 0.9337 | 0.8777 | 0.9331 | 0.8342 | 0.9353 | 0.8761 | 0.9350 | 0.7994 | 0.9327 |
| B0-aug | 0.9083 | 0.9355 | 0.9167 | 0.9359 | 0.9022 | 0.9338 | 0.8676 | 0.9354 | 0.9208 | 0.9370 | 0.8841 | 0.9347 |
| C | 0.9019 | 0.9333 | 0.9109 | 0.9342 | 0.8899 | 0.9317 | 0.8503 | 0.9327 | 0.9186 | 0.9346 | 0.8798 | 0.9320 |
| D | 0.9098 | 0.9344 | 0.9132 | 0.9350 | 0.8972 | 0.9320 | 0.8612 | 0.9345 | 0.9190 | 0.9355 | 0.8831 | 0.9328 |
| E | 0.9097 | 0.9347 | 0.9129 | 0.9349 | 0.9008 | 0.9323 | 0.8690 | 0.9343 | 0.9169 | 0.9357 | 0.8861 | 0.9332 |
| F | 0.9148 | 0.9354 | 0.9174 | 0.9353 | 0.9039 | 0.9326 | 0.8760 | 0.9347 | 0.9209 | 0.9356 | 0.8892 | 0.9337 |
| E-clean | 0.8218 | 0.9312 | 0.7577 | 0.9310 | 0.8602 | 0.9300 | 0.7903 | 0.9328 | 0.8286 | 0.9316 | 0.7585 | 0.9291 |

Figure: figures/perf_vs_snr.png (all four SNR levels, diagnostic only).

## 4. r diagnostics (fold 9, 3-seed mean r per record)

| model | clean | at -6 dB (all families, modes) | clean NORM | clean abnormal | Spearman(SNR, r), 4 levels |
|---|---|---|---|---|---|
| D | 0.3724 | 0.3292 | 0.3110 | 0.4217 | +0.136 |
| E | 0.3233 | 0.2701 | 0.2655 | 0.3696 | +0.185 |
| F | 0.3168 | 0.1673 | 0.2441 | 0.3751 | +0.392 |
| E-clean | 0.3717 | 0.4106 | 0.3143 | 0.4178 | -0.112 |

Mean r by family x SNR x mode, D (clean 0.3724):

| family | mode | +15 dB | +6 dB | +0 dB | -6 dB |
|---|---|---|---|---|---|
| baseline_wander | whole | 0.3670 | 0.3401 | 0.2896 | 0.2033 |
| baseline_wander | burst | 0.3710 | 0.3640 | 0.3505 | 0.3241 |
| emg | whole | 0.3687 | 0.3585 | 0.3541 | 0.3738 |
| emg | burst | 0.3712 | 0.3681 | 0.3674 | 0.3765 |
| motion_burst | whole | 0.3775 | 0.3912 | 0.3943 | 0.3560 |
| motion_burst | burst | 0.3794 | 0.4040 | 0.4272 | 0.4308 |
| dropout | whole | 0.3728 | 0.3620 | 0.3197 | 0.2813 |
| dropout | burst | 0.3742 | 0.3732 | 0.3580 | 0.3441 |
| powerline | whole | 0.3684 | 0.3525 | 0.3253 | 0.2764 |
| powerline | burst | 0.3711 | 0.3658 | 0.3571 | 0.3416 |
| mixed | whole | 0.3714 | 0.3616 | 0.3334 | 0.2787 |
| mixed | burst | 0.3736 | 0.3758 | 0.3758 | 0.3633 |

Mean r by family x SNR x mode, E (clean 0.3233):

| family | mode | +15 dB | +6 dB | +0 dB | -6 dB |
|---|---|---|---|---|---|
| baseline_wander | whole | 0.3182 | 0.2921 | 0.2434 | 0.1629 |
| baseline_wander | burst | 0.3219 | 0.3148 | 0.3011 | 0.2758 |
| emg | whole | 0.3200 | 0.3070 | 0.2913 | 0.3025 |
| emg | burst | 0.3222 | 0.3181 | 0.3136 | 0.3192 |
| motion_burst | whole | 0.3266 | 0.3333 | 0.3294 | 0.2853 |
| motion_burst | burst | 0.3287 | 0.3482 | 0.3676 | 0.3688 |
| dropout | whole | 0.3231 | 0.3077 | 0.2565 | 0.2146 |
| dropout | burst | 0.3245 | 0.3220 | 0.3040 | 0.2885 |
| powerline | whole | 0.3200 | 0.3035 | 0.2683 | 0.2121 |
| powerline | burst | 0.3222 | 0.3167 | 0.3051 | 0.2862 |
| mixed | whole | 0.3220 | 0.3096 | 0.2746 | 0.2180 |
| mixed | burst | 0.3242 | 0.3246 | 0.3212 | 0.3073 |

Mean r by family x SNR x mode, F (clean 0.3168):

| family | mode | +15 dB | +6 dB | +0 dB | -6 dB |
|---|---|---|---|---|---|
| baseline_wander | whole | 0.2899 | 0.1998 | 0.1155 | 0.0523 |
| baseline_wander | burst | 0.3081 | 0.2787 | 0.2491 | 0.2216 |
| emg | whole | 0.2895 | 0.2036 | 0.1354 | 0.0994 |
| emg | burst | 0.3076 | 0.2782 | 0.2533 | 0.2393 |
| motion_burst | whole | 0.3034 | 0.2652 | 0.2221 | 0.1574 |
| motion_burst | burst | 0.3182 | 0.3244 | 0.3320 | 0.3248 |
| dropout | whole | 0.2983 | 0.2308 | 0.1265 | 0.0823 |
| dropout | burst | 0.3115 | 0.2889 | 0.2537 | 0.2333 |
| powerline | whole | 0.2913 | 0.2002 | 0.1118 | 0.0612 |
| powerline | burst | 0.3081 | 0.2762 | 0.2418 | 0.2169 |
| mixed | whole | 0.2943 | 0.2173 | 0.1342 | 0.0738 |
| mixed | burst | 0.3110 | 0.2893 | 0.2668 | 0.2448 |

Mean r by family x SNR x mode, E-clean (clean 0.3717):

| family | mode | +15 dB | +6 dB | +0 dB | -6 dB |
|---|---|---|---|---|---|
| baseline_wander | whole | 0.3689 | 0.3504 | 0.3233 | 0.2598 |
| baseline_wander | burst | 0.3711 | 0.3670 | 0.3611 | 0.3427 |
| emg | whole | 0.3767 | 0.4184 | 0.5581 | 0.6811 |
| emg | burst | 0.3733 | 0.3882 | 0.4403 | 0.4926 |
| motion_burst | whole | 0.3770 | 0.3891 | 0.3946 | 0.3454 |
| motion_burst | burst | 0.3779 | 0.3992 | 0.4205 | 0.4137 |
| dropout | whole | 0.3737 | 0.3705 | 0.3484 | 0.3160 |
| dropout | burst | 0.3737 | 0.3760 | 0.3677 | 0.3564 |
| powerline | whole | 0.3732 | 0.3837 | 0.4340 | 0.4846 |
| powerline | burst | 0.3723 | 0.3763 | 0.3950 | 0.4177 |
| mixed | whole | 0.3743 | 0.3812 | 0.4068 | 0.4142 |
| mixed | burst | 0.3739 | 0.3810 | 0.3981 | 0.4031 |

Figure: figures/r_vs_snr.png.

## 5. Verdict (fold 9)

Best baseline (3-seed average): mixed = B0-aug (0.9094), unseen_families = B0-aug (0.9161).

| rule | E | F |
|---|---|---|
| 1. mixed: diff, 95% CI (need lo > 0, diff >= +0.005) | +0.0003 [-0.0018, +0.0027] FAIL | +0.0021 [+0.0000, +0.0044] FAIL |
| 2. unseen: diff, 95% CI (need diff >= +0.003, lo > -0.002) | -0.0013 [-0.0031, +0.0007] FAIL | +0.0012 [-0.0006, +0.0032] FAIL |
| 3. r clean - r at -6 dB (need >= 0.05); Spearman (need >= +0.3) | +0.0532; +0.185 FAIL | +0.1496; +0.392 pass |
| 4. r abnormal - r NORM (need >= -0.05) | +0.1041 pass | +0.1310 pass |
| 5. clean - B0-aug clean (need >= -0.003) | -0.0008 pass | -0.0009 pass |
| all rules | **NO-GO** | **NO-GO** |

**Fold-9 verdict: NO-GO** - no candidate passes all rules; fold 10 is not scored; findings go to the benchmark-and-analysis paper.

