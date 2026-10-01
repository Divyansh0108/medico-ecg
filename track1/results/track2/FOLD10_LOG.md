# Fold-10 evaluation log

One row = one entry (single model or ensemble) scored on fold 10. Rows 1-8 are from before Track 2,
reconstructed from the result files. Every later row is written by fold10_log.log_fold10() when the
scoring happens.

## Disclosure (written 2026-10-01, before the Track 2 fold-10 reporting pass)

Before this pass, fold 10 had been scored for 12 entries. An entry is one model or one ensemble. The
per-member fold-10 scores printed while an ensemble was scored are not counted separately: 5 members in
round 2 and 15 in round 3.
- Selection-related entries: 8 (#1-8, Track 1 rounds 1-3). #1-2 (round 1, M1/M2 full length) fed the
  pre-agreed round-1 gate in decide.py ("continue with seeds 1-4 iff the best seed-0 TEST macro-AUROC
  >= 0.92"). The gate failed, and that project-level decision (start round 2 instead of more seeds)
  DID depend on fold 10. No model or hyperparameter was chosen from those scores. #3-8 were scored after
  the configurations were fixed on fold 9 (results/crop/SELECTION.md, SELECTION2.md); those files record
  that the choices used fold 9 only. #3 (M1-crop) was requested and was scored before SELECTION.md (file times 10:26 vs 12:37)
  was written.
- Calibration entries: 4 (#9-12, Track 2 step 2). Platt scaling and the thresholds were fitted on fold 9;
  fold 10 was only used to report ECE/F1 after fitting.
- No Track 2 choice (severity set, variants, training recipe, rules, verdict) and no choice in RULES2.md
  depended on any fold-10 result. Apart from the round-1 gate above, no model or hyperparameter choice
  in the project used fold 10.
Rows from #13 on are the single descriptive reporting pass of RULES2.md section 4 (no selection).

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
