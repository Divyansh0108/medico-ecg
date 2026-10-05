# PTB-XL Track 2 - calibration, corruption benchmark, reliability-gated variants (2026-10-01)

Fold 9 for all selection. The decision rules were committed to git (`4c77715`) before any Track 2 run.
Full tables: `../results/robustness/REPORT.md`. Rules: `../results/robustness/RULES.md`. Every fold-10 use:
`../results/robustness/FOLD10_LOG.md` (12 entries in total; none for the Track 2 variants).

## 1. Calibration (resnet1d_wang seeds 0-2, no retraining; fitted on fold 9, scored on fold 10)

| entry | fold-10 mAUROC | 95% CI | ECE before -> after | macro-F1 | bal. acc. |
|---|---|---|---|---|---|
| seed 0 | 0.9332 | 0.926-0.940 | 0.117 -> 0.019 | 0.759 | 0.853 |
| seed 1 | 0.9307 | 0.923-0.938 | 0.122 -> 0.019 | 0.761 | 0.846 |
| seed 2 | 0.9311 | 0.924-0.938 | 0.119 -> 0.019 | 0.751 | 0.847 |
| 3-seed average | 0.9347 | 0.927-0.941 | 0.120 -> 0.019 | 0.764 | 0.849 |

The raw probabilities are over-confident for positives (pos_weight + mixup), worst for HYP. Per-class
Platt scaling fixes this, and AUROC is unchanged.

## 2. Corruption benchmark (`noise/synthetic.py`, 53 unit tests)

- Noise is added to the band-passed signal before dataset standardization, so it is not filtered and
  not z-scored away.
- 12 leads, each lead scaled to its own SNR.
- Families: baseline wander and EMG (seen in training); motion burst, dropout and powerline (unseen);
  mixed (2 of the 5).
- Modes: whole record, or a 2-4 s burst.
- Every model sees the same seeded noise.
- Two definitions to check: powerline at 100 Hz sits at Nyquist, so the mains frequency drifts in
  49.9-50.1 Hz. Dropout is flat-line gaps at a random offset, so its SNR can be controlled.

Difficulty for the clean-trained model, mean drop on mixed: 15 dB 0.000, 6 dB 0.006, 0 dB 0.023,
-6 dB 0.079. Rule: bar 0.03 -> **severity set {-6 dB}**. Burst mode barely hurts at any SNR (<= 0.008);
the difficulty comes from whole-record noise.

## 3. Variants (resnet1d_wang trunk split after stage 1; params within 1% of B0; seeds 0-2)

C concat, D generic gate, E reliability gate r, F = E + consistency (>= 15 dB pairs only) + severity
ranking on r, E-clean = E without augmentation.

Fold 9, -6 dB, 3-seed probability average:

| model | clean | seen | unseen | mixed | all corrupted |
|---|---|---|---|---|---|
| B0-clean | 0.9389 | 0.8947 | 0.8985 | 0.8661 | 0.8919 |
| B0-aug | 0.9389 | 0.9241 | 0.9161 | 0.9094 | 0.9177 |
| C | 0.9371 | 0.9201 | 0.9096 | 0.9059 | 0.9125 |
| D | 0.9380 | 0.9231 | 0.9132 | 0.9080 | 0.9156 |
| E | 0.9381 | 0.9231 | 0.9148 | 0.9097 | 0.9167 |
| F | 0.9380 | 0.9257 | 0.9173 | 0.9115 | 0.9191 |
| E-clean | 0.9376 | 0.8604 | 0.8789 | 0.8438 | 0.8669 |

## 4. Verdict: NO-GO (fold 9; best baseline = B0-aug on both mixed and unseen)

| rule | E | F |
|---|---|---|
| 1. mixed diff (need >= +0.005, CI lo > 0) | +0.0003 [-0.0018, +0.0027] fail | +0.0021 [+0.0000, +0.0044] fail |
| 2. unseen diff (need >= +0.003) | -0.0013 fail | +0.0012 fail |
| 3. r drop at -6 dB >= 0.05 and Spearman >= 0.3 | 0.053; 0.19 fail | 0.150; 0.39 pass |
| 4. abnormal r >= NORM r - 0.05 | pass | pass |
| 5. clean within 0.003 of B0-aug | pass | pass |

Fold 10 was not scored for any variant, as the rules require.

## Findings for the benchmark-and-analysis paper

- Augmentation does most of the work: mixed at -6 dB goes from 0.866 to 0.909. The trunk variants add
  at most +0.002 on top.
- F's r drops steadily with severity for every family except motion bursts, and it is not lower on
  abnormal records. The gate behaves as intended, but that did not give enough accuracy gain.
- Without augmentation the gate is not a reliability signal: E-clean's r rises under EMG and powerline.
- Seed spread on corrupted groups is 0.001-0.006 (E-clean up to 0.017), larger than most variant gaps.

## Notes

- The severity set has one level, so the r-vs-SNR Spearman would be undefined. An amendment written
  before any variant run computes r diagnostics over all four SNR levels; performance rules use -6 dB.
- All runs early-stop on clean fold-9 macro-AUROC.
- Fold-10 count correction: the round-3 summary first said 7 entries; the correct count is 8 (now 12).

## Files

- `noise/synthetic.py`, `tests/test_corruptions.py`, `models/gated.py`, `training/train.py` (`--aug`, `--aux`)
- `evaluation/robustness_eval.py`, `evaluation/difficulty_check.py`, `reporting/primary_verdict_report.py`, `evaluation/robustness_calibration.py`, `evaluation/fold10_log.py`
- `scripts/rules1_train.sh`, `scripts/rules1_eval.sh`
- `../results/robustness/`: REPORT, RULES, LOG, FOLD10_LOG, calibration/difficulty/groups/verdict JSON,
  `figures/`, `grid/` (all predictions and r)
- `checkpoints/`: all weights
