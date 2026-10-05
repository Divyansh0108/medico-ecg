# RACER Phase 1 — summary

## Verdict: **NO-GO / PIVOT**

Augmented baselines match RACER-lite (E) on at least one binding criterion, or r does not track severity, or r drops on clean-abnormal ECGs. **The outcome supports a benchmark-and-analysis paper instead.**

## B8 criteria

| candidate | criterion | value | rule | pass | binding |
|---|---|---|---|---|---|
| E | AUROC diff vs best aug baseline (A) on mixed | +0.0001 [-0.0024, +0.0025] | 95% CI lower bound > 0 | FAIL | yes |
| E | mean r monotone in severity | 0.60836 > 0.60641 > 0.60370 | r(15 dB) > r(6 dB) > r(0 dB) | PASS | yes |
| E | r on clean-abnormal vs clean-NORM | 0.58803 vs 0.63610 | r_abn >= r_norm - 0.05 | PASS | yes |
| E | AUROC diff vs best aug baseline (A) on mixed (severe) | -0.0004 [-0.0030, +0.0021] | 95% CI lower bound > 0 | FAIL | no |
| E | AUROC diff vs best aug baseline (A) on unseen_families (all) | -0.0001 [-0.0026, +0.0023] | 95% CI lower bound > 0 | FAIL | no |
| E | AUROC diff vs best aug baseline (A) on unseen_families (severe) | -0.0005 [-0.0030, +0.0021] | 95% CI lower bound > 0 | FAIL | no |

Operationalization (fixed before seeing results):
- Candidates: E trained in regime aug. Baselines: augmented A, C. The *best* baseline for a group is the one with the highest seed-averaged-probability macro-AUROC on that group.
- Binding groups: mixed, pooled over severities and injection modes (group AUROC = unweighted mean over its conditions). 0 dB and unseen-family comparisons are reported as supporting (non-binding) evidence.
- CI: paired bootstrap over test patients (1000 resamples of 1877 patient clusters, 2158 records), 95% percentile interval, on macro-AUROC of seed-averaged predicted probabilities.
- r monotonicity: mean r of the candidate over all corrupted test conditions at each severity (all families incl. mixed, both modes), averaged over seeds; strict mild > moderate > severe.
- Clean-abnormal = no NORM label; clean-NORM = NORM as the only label; pass if r_abn >= r_norm - 0.05.
- GO iff one candidate (E) passes all binding criteria.

## Macro-AUROC (mean ± std over seeds)

Group value = unweighted mean over its conditions; drop = clean − corrupted.

| variant | regime | clean | mixed (all) | mixed (severe) | seen_families (all) | unseen_families (all) | all_corrupted (all) |
|---|---|---|---|---|---|---|---|
| A | aug | 0.8496 ± 0.0013 | 0.8452 ± 0.0015 | 0.8415 ± 0.0017 | 0.8488 ± 0.0013 | 0.8458 ± 0.0014 | 0.8470 ± 0.0014 |
| A | clean | 0.8484 ± 0.0033 | 0.8362 ± 0.0026 | 0.8251 ± 0.0037 | 0.8382 ± 0.0024 | 0.8448 ± 0.0035 | 0.8400 ± 0.0026 |
| C | aug | 0.8478 ± 0.0008 | 0.8438 ± 0.0009 | 0.8399 ± 0.0016 | 0.8466 ± 0.0010 | 0.8444 ± 0.0010 | 0.8452 ± 0.0009 |
| C | clean | 0.8481 ± 0.0003 | 0.8341 ± 0.0030 | 0.8214 ± 0.0066 | 0.8382 ± 0.0033 | 0.8445 ± 0.0007 | 0.8394 ± 0.0022 |
| E | aug | 0.8496 ± 0.0023 | 0.8454 ± 0.0021 | 0.8414 ± 0.0025 | 0.8485 ± 0.0023 | 0.8459 ± 0.0023 | 0.8469 ± 0.0022 |
| E | clean | 0.8502 ± 0.0015 | 0.8342 ± 0.0044 | 0.8185 ± 0.0081 | 0.8394 ± 0.0029 | 0.8466 ± 0.0020 | 0.8406 ± 0.0027 |

Drop in macro-AUROC (clean − corrupted), mean ± std over seeds:

| variant | regime | mixed (all) | mixed (severe) | seen_families (all) | unseen_families (all) | all_corrupted (all) |
|---|---|---|---|---|---|---|
| A | aug | 0.0045 ± 0.0003 | 0.0081 ± 0.0005 | 0.0009 ± 0.0003 | 0.0039 ± 0.0006 | 0.0027 ± 0.0002 |
| A | clean | 0.0122 ± 0.0024 | 0.0233 ± 0.0052 | 0.0101 ± 0.0026 | 0.0035 ± 0.0003 | 0.0084 ± 0.0017 |
| C | aug | 0.0039 ± 0.0007 | 0.0078 ± 0.0014 | 0.0011 ± 0.0004 | 0.0034 ± 0.0009 | 0.0025 ± 0.0005 |
| C | clean | 0.0141 ± 0.0030 | 0.0268 ± 0.0065 | 0.0100 ± 0.0033 | 0.0036 ± 0.0007 | 0.0088 ± 0.0022 |
| E | aug | 0.0042 ± 0.0007 | 0.0082 ± 0.0014 | 0.0012 ± 0.0004 | 0.0037 ± 0.0005 | 0.0027 ± 0.0004 |
| E | clean | 0.0160 ± 0.0035 | 0.0317 ± 0.0078 | 0.0108 ± 0.0022 | 0.0036 ± 0.0007 | 0.0095 ± 0.0018 |

## Paired bootstrap vs best augmented baseline

| candidate | group | severity | best baseline | diff | 95% CI |
|---|---|---|---|---|---|
| E | seen_families | mild | A | -0.0002 | [-0.0027, +0.0022] |
| E | seen_families | moderate | A | -0.0004 | [-0.0029, +0.0020] |
| E | seen_families | severe | A | -0.0007 | [-0.0033, +0.0016] |
| E | seen_families | all | A | -0.0004 | [-0.0029, +0.0020] |
| E | unseen_families | mild | A | +0.0001 | [-0.0024, +0.0026] |
| E | unseen_families | moderate | A | -0.0000 | [-0.0025, +0.0025] |
| E | unseen_families | severe | A | -0.0005 | [-0.0030, +0.0021] |
| E | unseen_families | all | A | -0.0001 | [-0.0026, +0.0023] |
| E | mixed | mild | A | +0.0003 | [-0.0022, +0.0027] |
| E | mixed | moderate | A | +0.0003 | [-0.0021, +0.0027] |
| E | mixed | severe | A | -0.0004 | [-0.0030, +0.0021] |
| E | mixed | all | A | +0.0001 | [-0.0024, +0.0025] |
| E | all_corrupted | mild | A | +0.0000 | [-0.0025, +0.0025] |
| E | all_corrupted | moderate | A | -0.0001 | [-0.0025, +0.0024] |
| E | all_corrupted | severe | A | -0.0006 | [-0.0031, +0.0018] |
| E | all_corrupted | all | A | -0.0002 | [-0.0026, +0.0022] |

All pairwise comparisons: `bootstrap.csv`.

## Reliability r (mean over seeds)

| variant | regime | clean-NORM | clean-abnormal | mild | moderate | severe |
|---|---|---|---|---|---|---|
| E | aug | 0.6361 | 0.5880 | 0.6084 | 0.6064 | 0.6037 |
| E | clean | 0.6473 | 0.5982 | 0.6189 | 0.6153 | 0.6076 |

Per-family r vs severity: `reliability.csv`, `figures/r_vs_snr.png`.

## Notes

- Data: PTB-XL, lead II at 100 Hz, test fold(s) [10]. Training corruption families: baseline_wander, emg; unseen at test: motion_burst, dropout.
- No real (NSTDB) noise in this phase: all corruptions are synthetic.
- 'mixed' test condition: per record, 2 distinct families drawn from {baseline_wander, emg, motion_burst} (all configured additive families), summed at unit power and scaled to the target SNR; × 3 severities × 2 modes.

## Runs

| run | params | device | amp | best epoch | val AUROC | git |
|---|---|---|---|---|---|---|
| A_aug_s0 | 1,235,237 | mps | float32 | 11 | 0.8547 | nogit* |
| A_aug_s1 | 1,235,237 | mps | float32 | 14 | 0.8527 | nogit* |
| A_aug_s2 | 1,235,237 | mps | float32 | 16 | 0.8520 | nogit* |
| A_aug_s3 | 1,235,237 | mps | float32 | 16 | 0.8531 | nogit* |
| A_aug_s4 | 1,235,237 | mps | float32 | 13 | 0.8558 | nogit* |
| A_clean_s0 | 1,235,237 | mps | float32 | 11 | 0.8550 | nogit* |
| A_clean_s1 | 1,235,237 | mps | float32 | 14 | 0.8524 | nogit* |
| A_clean_s2 | 1,235,237 | mps | float32 | 8 | 0.8538 | nogit* |
| A_clean_s3 | 1,235,237 | mps | float32 | 6 | 0.8519 | nogit* |
| A_clean_s4 | 1,235,237 | mps | float32 | 9 | 0.8541 | nogit* |
| C_aug_s0 | 1,093,189 | mps | float32 | 14 | 0.8543 | nogit* |
| C_aug_s1 | 1,093,189 | mps | float32 | 14 | 0.8504 | nogit* |
| C_aug_s2 | 1,093,189 | mps | float32 | 8 | 0.8545 | nogit* |
| C_aug_s3 | 1,093,189 | mps | float32 | 10 | 0.8518 | nogit* |
| C_aug_s4 | 1,093,189 | mps | float32 | 13 | 0.8501 | nogit* |
| C_clean_s0 | 1,093,189 | mps | float32 | 14 | 0.8546 | nogit* |
| C_clean_s1 | 1,093,189 | mps | float32 | 17 | 0.8518 | nogit* |
| C_clean_s2 | 1,093,189 | mps | float32 | 8 | 0.8546 | nogit* |
| C_clean_s3 | 1,093,189 | mps | float32 | 10 | 0.8522 | nogit* |
| C_clean_s4 | 1,093,189 | mps | float32 | 13 | 0.8522 | nogit* |
| E_aug_s0 | 1,174,853 | mps | float32 | 14 | 0.8535 | nogit* |
| E_aug_s1 | 1,174,853 | mps | float32 | 14 | 0.8504 | nogit* |
| E_aug_s2 | 1,174,853 | mps | float32 | 8 | 0.8545 | nogit* |
| E_aug_s3 | 1,174,853 | mps | float32 | 13 | 0.8523 | nogit* |
| E_aug_s4 | 1,174,853 | mps | float32 | 13 | 0.8525 | nogit* |
| E_clean_s0 | 1,174,853 | mps | float32 | 14 | 0.8545 | nogit* |
| E_clean_s1 | 1,174,853 | mps | float32 | 15 | 0.8539 | nogit* |
| E_clean_s2 | 1,174,853 | mps | float32 | 8 | 0.8553 | nogit* |
| E_clean_s3 | 1,174,853 | mps | float32 | 9 | 0.8523 | nogit* |
| E_clean_s4 | 1,174,853 | mps | float32 | 11 | 0.8535 | nogit* |
