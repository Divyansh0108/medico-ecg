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
