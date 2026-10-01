# NSTDB real-noise benchmark (fold 9)

Rules: RULES2.md section 1 (committed before this run). Descriptive; it does not change the Track 2 verdict. Fold 10 is not used here.

## Setup

- MIT-BIH NSTDB records bw, ma, em: 2 channels, 360 Hz, 650,000 samples (30.09 min) each; resampled to 100 Hz (polyphase 5/18) -> 180556 samples, then band-passed 0.5-40 Hz with the ECG filter.
- Time split per record (100 Hz samples): TRAIN (0, 108333), 10 s gap, EVAL (109333, 180556) (asserted disjoint; tests/test_nstdb.py). Only EVAL noise is used here.
- Per 10 s fold-9 record: one random excerpt; each of the 12 leads takes one of the 2 noise channels at random; each lead's noise is scaled to the target SNR against that lead's power. nstdb_mixed = two distinct families, each unit power, summed and rescaled. Modes whole (10 s) and burst (2-4 s). Noise is added before dataset-level standardization. Fixed seeded RNG per (condition, record).
- Models: existing checkpoints, seeds 0-2, 3-seed probability average unless stated. Group = mean macro-AUROC over its conditions; per-family groups pool SNR {0, -6} dB and both modes.

## Scores per family (3-seed average; drop vs the model's own clean score in brackets)

| model | clean | nstdb_bw | nstdb_ma | nstdb_em | nstdb_mixed | nstdb_all |
|---|---|---|---|---|---|---|
| B0-clean | 0.9389 | 0.8928 (+0.046) | 0.8978 (+0.041) | 0.8819 (+0.057) | 0.8843 (+0.055) | 0.8892 (+0.050) |
| B0-aug | 0.9389 | 0.9082 (+0.031) | 0.9144 (+0.024) | 0.8964 (+0.042) | 0.9023 (+0.037) | 0.9053 (+0.034) |
| C | 0.9371 | 0.8981 (+0.039) | 0.8987 (+0.038) | 0.8805 (+0.057) | 0.8892 (+0.048) | 0.8916 (+0.046) |
| D | 0.9380 | 0.9032 (+0.035) | 0.8997 (+0.038) | 0.8863 (+0.052) | 0.8934 (+0.045) | 0.8957 (+0.042) |
| E | 0.9381 | 0.9038 (+0.034) | 0.8994 (+0.039) | 0.8862 (+0.052) | 0.8928 (+0.045) | 0.8956 (+0.043) |
| F | 0.9380 | 0.9073 (+0.031) | 0.9050 (+0.033) | 0.8893 (+0.049) | 0.8966 (+0.041) | 0.8996 (+0.038) |
| E-clean | 0.9376 | 0.8811 (+0.056) | 0.8705 (+0.067) | 0.8654 (+0.072) | 0.8651 (+0.072) | 0.8705 (+0.067) |

At +0 dB only (both modes):

| model | nstdb_bw | nstdb_ma | nstdb_em | nstdb_mixed | nstdb_all |
|---|---|---|---|---|---|
| B0-clean | 0.9149 (+0.024) | 0.9193 (+0.020) | 0.9095 (+0.029) | 0.9103 (+0.029) | 0.9135 (+0.025) |
| B0-aug | 0.9227 (+0.016) | 0.9269 (+0.012) | 0.9152 (+0.024) | 0.9199 (+0.019) | 0.9212 (+0.018) |
| C | 0.9200 (+0.017) | 0.9214 (+0.016) | 0.9094 (+0.028) | 0.9153 (+0.022) | 0.9165 (+0.021) |
| D | 0.9203 (+0.018) | 0.9210 (+0.017) | 0.9106 (+0.027) | 0.9149 (+0.023) | 0.9167 (+0.021) |
| E | 0.9216 (+0.017) | 0.9215 (+0.017) | 0.9115 (+0.027) | 0.9157 (+0.022) | 0.9176 (+0.021) |
| F | 0.9234 (+0.015) | 0.9240 (+0.014) | 0.9123 (+0.026) | 0.9173 (+0.021) | 0.9193 (+0.019) |
| E-clean | 0.9115 (+0.026) | 0.9035 (+0.034) | 0.9012 (+0.036) | 0.8998 (+0.038) | 0.9040 (+0.034) |

At -6 dB only (both modes):

| model | nstdb_bw | nstdb_ma | nstdb_em | nstdb_mixed | nstdb_all |
|---|---|---|---|---|---|
| B0-clean | 0.8706 (+0.068) | 0.8764 (+0.062) | 0.8543 (+0.085) | 0.8583 (+0.081) | 0.8649 (+0.074) |
| B0-aug | 0.8936 (+0.045) | 0.9019 (+0.037) | 0.8776 (+0.061) | 0.8847 (+0.054) | 0.8895 (+0.049) |
| C | 0.8761 (+0.061) | 0.8759 (+0.061) | 0.8516 (+0.086) | 0.8631 (+0.074) | 0.8667 (+0.070) |
| D | 0.8861 (+0.052) | 0.8784 (+0.060) | 0.8621 (+0.076) | 0.8719 (+0.066) | 0.8746 (+0.063) |
| E | 0.8861 (+0.052) | 0.8773 (+0.061) | 0.8609 (+0.077) | 0.8699 (+0.068) | 0.8735 (+0.065) |
| F | 0.8912 (+0.047) | 0.8861 (+0.052) | 0.8662 (+0.072) | 0.8760 (+0.062) | 0.8799 (+0.058) |
| E-clean | 0.8507 (+0.087) | 0.8375 (+0.100) | 0.8296 (+0.108) | 0.8305 (+0.107) | 0.8371 (+0.101) |

Per condition (3-seed average macro-AUROC):

| model | bw +0 whole | bw +0 burst | bw -6 whole | bw -6 burst | ma +0 whole | ma +0 burst | ma -6 whole | ma -6 burst | em +0 whole | em +0 burst | em -6 whole | em -6 burst | mixed +0 whole | mixed +0 burst | mixed -6 whole | mixed -6 burst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0-clean | 0.8961 | 0.9336 | 0.8087 | 0.9326 | 0.9031 | 0.9354 | 0.8201 | 0.9328 | 0.8848 | 0.9343 | 0.7762 | 0.9324 | 0.8867 | 0.9339 | 0.7845 | 0.9322 |
| B0-aug | 0.9115 | 0.9340 | 0.8549 | 0.9324 | 0.9182 | 0.9357 | 0.8707 | 0.9332 | 0.8964 | 0.9341 | 0.8231 | 0.9321 | 0.9057 | 0.9340 | 0.8367 | 0.9328 |
| C | 0.9076 | 0.9324 | 0.8227 | 0.9295 | 0.9096 | 0.9332 | 0.8226 | 0.9293 | 0.8868 | 0.9321 | 0.7747 | 0.9284 | 0.8977 | 0.9328 | 0.7967 | 0.9295 |
| D | 0.9082 | 0.9324 | 0.8417 | 0.9304 | 0.9082 | 0.9338 | 0.8262 | 0.9305 | 0.8888 | 0.9323 | 0.7945 | 0.9298 | 0.8972 | 0.9326 | 0.8138 | 0.9300 |
| E | 0.9100 | 0.9331 | 0.8415 | 0.9306 | 0.9088 | 0.9343 | 0.8250 | 0.9296 | 0.8905 | 0.9324 | 0.7927 | 0.9291 | 0.8988 | 0.9326 | 0.8104 | 0.9293 |
| F | 0.9132 | 0.9337 | 0.8510 | 0.9313 | 0.9135 | 0.9346 | 0.8417 | 0.9305 | 0.8915 | 0.9331 | 0.8025 | 0.9300 | 0.9011 | 0.9335 | 0.8215 | 0.9305 |
| E-clean | 0.8905 | 0.9324 | 0.7714 | 0.9300 | 0.8737 | 0.9332 | 0.7464 | 0.9285 | 0.8698 | 0.9326 | 0.7307 | 0.9285 | 0.8674 | 0.9322 | 0.7330 | 0.9279 |

Per seed (0-2), mean +- std:

| model | clean | nstdb_all | nstdb_all at -6 dB |
|---|---|---|---|
| B0-clean | 0.9356 +- 0.0008 | 0.8813 +- 0.0033 | 0.8554 +- 0.0045 |
| B0-aug | 0.9361 +- 0.0016 | 0.8997 +- 0.0061 | 0.8826 +- 0.0079 |
| C | 0.9339 +- 0.0013 | 0.8847 +- 0.0059 | 0.8579 +- 0.0094 |
| D | 0.9346 +- 0.0011 | 0.8883 +- 0.0015 | 0.8653 +- 0.0022 |
| E | 0.9347 +- 0.0014 | 0.8885 +- 0.0016 | 0.8649 +- 0.0010 |
| F | 0.9348 +- 0.0017 | 0.8930 +- 0.0007 | 0.8717 +- 0.0012 |
| E-clean | 0.9340 +- 0.0013 | 0.8599 +- 0.0047 | 0.8241 +- 0.0066 |

For reference, synthetic benchmark at -6 dB (REPORT.md, 3-seed average): B0-clean all_corrupted 0.8919 / mixed 0.8661, B0-aug all_corrupted 0.9177 / mixed 0.9094, C all_corrupted 0.9125 / mixed 0.9059, D all_corrupted 0.9156 / mixed 0.9080, E all_corrupted 0.9167 / mixed 0.9097, F all_corrupted 0.9191 / mixed 0.9115, E-clean all_corrupted 0.8669 / mixed 0.8438.

## Paired patient bootstrap at -6 dB (1000 resamples, seed 0)

| comparison | nstdb_all @ -6 dB | nstdb_mixed @ -6 dB |
|---|---|---|
| F - B0-aug | -0.0096 [-0.0117, -0.0074] | -0.0088 [-0.0115, -0.0059] |
| E - B0-aug | -0.0160 [-0.0182, -0.0138] | -0.0149 [-0.0177, -0.0119] |
| D - B0-aug | -0.0149 [-0.0168, -0.0129] | -0.0128 [-0.0155, -0.0101] |
| C - B0-aug | -0.0228 [-0.0253, -0.0204] | -0.0216 [-0.0247, -0.0187] |
| B0-aug - B0-clean | +0.0246 [+0.0225, +0.0267] | +0.0264 [+0.0234, +0.0296] |

Rule (descriptive, RULES2.md 1): claim "F beats B0-aug on real noise" only if on nstdb_all at -6 dB the CI lower bound > 0 and the difference >= +0.005. Observed -0.0096 [-0.0117, -0.0074] -> **claim NOT supported**.
(nstdb_mixed at -6 dB, reported only: -0.0088 [-0.0115, -0.0059].)

## r diagnostics (3-seed mean r per record)

| model | r clean | r nstdb_all at 0 dB | r nstdb_all at -6 dB | Spearman(SNR, r), 0 and -6 dB | Spearman incl. clean | Spearman nstdb_bw | Spearman nstdb_ma | Spearman nstdb_em | Spearman nstdb_mixed |
|---|---|---|---|---|---|---|---|---|---|
| D | 0.3724 | 0.4565 | 0.4870 | -0.131 | -0.201 | -0.091 | -0.222 | -0.086 | -0.148 |
| E | 0.3233 | 0.3885 | 0.4206 | -0.137 | -0.195 | -0.082 | -0.253 | -0.076 | -0.168 |
| F | 0.3168 | 0.3440 | 0.3647 | -0.086 | -0.110 | -0.030 | -0.179 | -0.045 | -0.102 |
| E-clean | 0.3717 | 0.4778 | 0.5045 | -0.099 | -0.188 | -0.065 | -0.241 | -0.018 | -0.122 |

Spearman(SNR, r) pairs each record's r with the condition SNR over all 16 NSTDB conditions (positive = r rises with SNR). 'incl. clean' adds the clean records as the highest SNR level.

Mean r by family x SNR x mode:

| model | clean | bw +0 whole | bw +0 burst | bw -6 whole | bw -6 burst | ma +0 whole | ma +0 burst | ma -6 whole | ma -6 burst | em +0 whole | em +0 burst | em -6 whole | em -6 burst | mixed +0 whole | mixed +0 burst | mixed -6 whole | mixed -6 burst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D | 0.3724 | 0.4329 | 0.4003 | 0.4570 | 0.4111 | 0.5255 | 0.4348 | 0.6009 | 0.4682 | 0.5098 | 0.4226 | 0.5328 | 0.4356 | 0.5025 | 0.4238 | 0.5460 | 0.4441 |
| E | 0.3233 | 0.3577 | 0.3411 | 0.3761 | 0.3503 | 0.4559 | 0.3766 | 0.5440 | 0.4156 | 0.4272 | 0.3612 | 0.4464 | 0.3721 | 0.4246 | 0.3638 | 0.4742 | 0.3864 |
| F | 0.3168 | 0.2894 | 0.3140 | 0.2933 | 0.3166 | 0.3764 | 0.3417 | 0.4412 | 0.3707 | 0.3918 | 0.3448 | 0.4033 | 0.3531 | 0.3578 | 0.3360 | 0.3891 | 0.3506 |
| E-clean | 0.3717 | 0.4597 | 0.4095 | 0.4760 | 0.4182 | 0.5841 | 0.4621 | 0.6735 | 0.4980 | 0.5112 | 0.4228 | 0.5141 | 0.4288 | 0.5353 | 0.4374 | 0.5718 | 0.4555 |
