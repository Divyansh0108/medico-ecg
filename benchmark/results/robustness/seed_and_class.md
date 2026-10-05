# Per-seed spread and r by diagnostic class (fold 9, descriptive)

Stored fold-9 predictions (grid/main). Not part of any decision rule.

## 1. F vs B0-aug per seed at -6 dB

Group score = mean macro-AUROC over the group's conditions at -6 dB, both modes. '3-seed avg' = macro-AUROC of the averaged probabilities (the score used by the rules).

| model | seed | clean | seen_families | unseen_families | mixed | all_corrupted |
|---|---|---|---|---|---|---|
| B0-aug | 0 | 0.9379 | 0.9225 | 0.9136 | 0.9100 | 0.9160 |
| B0-aug | 1 | 0.9356 | 0.9182 | 0.9072 | 0.9016 | 0.9100 |
| B0-aug | 2 | 0.9347 | 0.9181 | 0.9088 | 0.9010 | 0.9106 |
| B0-aug | mean +- std | 0.9361 +- 0.0016 | 0.9196 +- 0.0025 | 0.9099 +- 0.0033 | 0.9042 +- 0.0050 | 0.9122 +- 0.0033 |
| B0-aug | **3-seed avg** | **0.9389** | **0.9241** | **0.9161** | **0.9094** | **0.9177** |
| F | 0 | 0.9361 | 0.9234 | 0.9062 | 0.9036 | 0.9115 |
| F | 1 | 0.9355 | 0.9204 | 0.9108 | 0.9050 | 0.9130 |
| F | 2 | 0.9329 | 0.9184 | 0.9084 | 0.9036 | 0.9109 |
| F | mean +- std | 0.9348 +- 0.0017 | 0.9207 +- 0.0025 | 0.9085 +- 0.0023 | 0.9041 +- 0.0008 | 0.9118 +- 0.0011 |
| F | **3-seed avg** | **0.9380** | **0.9257** | **0.9173** | **0.9115** | **0.9191** |

F - B0-aug:

| | clean | seen_families | unseen_families | mixed | all_corrupted |
|---|---|---|---|---|---|
| seed 0 (F_s0 - B0aug_s0) | -0.0017 | +0.0008 | -0.0074 | -0.0063 | -0.0045 |
| seed 1 (F_s1 - B0aug_s1) | -0.0002 | +0.0022 | +0.0036 | +0.0034 | +0.0031 |
| seed 2 (F_s2 - B0aug_s2) | -0.0018 | +0.0003 | -0.0004 | +0.0026 | +0.0003 |
| mean of per-seed differences | -0.0012 | +0.0011 | -0.0014 | -0.0001 | -0.0004 |
| 3-seed avg difference | -0.0009 | +0.0016 | +0.0012 | +0.0021 | +0.0015 |
| ensembling gain, B0-aug (avg - mean of seeds) | +0.0028 | +0.0045 | +0.0062 | +0.0052 | +0.0055 |
| ensembling gain, F | +0.0031 | +0.0050 | +0.0088 | +0.0074 | +0.0073 |

## 2. r by diagnostic class (clean fold-9 records, 3-seed mean r)

Superclass-only = records whose only superclass label is that class. non-NORM = records without the NORM label (any combination of MI, STTC, CD, HYP).

| model | NORM-only (n=914) | MI-only (n=233) | STTC-only (n=255) | CD-only (n=171) | HYP-only (n=64) | NORM (n=955) | non-NORM (n=1191) |
|---|---|---|---|---|---|---|---|
| D | 0.3080 | 0.4109 | 0.3722 | 0.3906 | 0.4058 | 0.3110 | 0.4217 |
| E | 0.2632 | 0.3584 | 0.3213 | 0.3341 | 0.3744 | 0.2655 | 0.3696 |
| F | 0.2414 | 0.3718 | 0.3017 | 0.3426 | 0.3662 | 0.2441 | 0.3751 |
| E-clean | 0.3116 | 0.4130 | 0.3697 | 0.3896 | 0.3959 | 0.3143 | 0.4178 |

Spearman(SNR, r) over all (record, condition) pairs of the corrupted conditions at +15, +6, 0, -6 dB (all six families, both modes), computed inside each record subset:

| model | all records (n=2146) | NORM-only (n=914) | non-NORM (n=1191) | mean r drop clean -> -6 dB, NORM-only | same, non-NORM |
|---|---|---|---|---|---|
| D | +0.136 | +0.094 | +0.191 | +0.0193 | +0.0614 |
| E | +0.185 | +0.157 | +0.233 | +0.0281 | +0.0724 |
| F | +0.392 | +0.430 | +0.415 | +0.1140 | +0.1768 |
