# Round 3 selection (seeds, SWA/EMA, label smoothing, crop 500) - rule written BEFORE any run, 2026-09-30

Base recipe: 2.5 s random crops, sliding-window mean (250/125), band-pass on, dataset-level norm fitted on
train, mixup 0.4, Adam 5e-4 cosine. All selection on fold 9 only.

## Rule (fixed before the runs)

- Noise floor: std (ddof 1) of fold-9 mAUROC over seeds 0-4 of resnet1d_wang_crop_dsnorm.
- FINAL2_single: seed-0 resnet1d_wang. The variants SWA, EMA, label smoothing 0.05 and crop 500/250 are each
  compared with the seed-0 base run (0.9365). The variant with the largest fold-9 gain replaces the base run
  only if its gain is larger than the noise floor; otherwise the base run stays. Variants are not combined
  (each was tested alone).
- FINAL2_ensemble: mean of probabilities of seeds 0-4 of resnet1d_wang, inception1d and xresnet1d50
  (base recipe, 15 models). No subset search, no weights.
- Baselines for the paired bootstrap: resnet1d_wang_crop_dsnorm and xresnet1d101_crop_dsnorm (seed 0,
  base recipe) = the published architectures reproduced in this pipeline.
- Fold 10: each final entry is evaluated once with final_eval.py, after the section below is filled in.

Already known about fold 10 (from round 2, seed 0): FINAL_single (resnet1d_wang_crop_dsnorm) 0.9332 and
FINAL_ensemble 0.9364, plus the five member scores. None of these is used by the rule above.

## Decision (filled in after the fold-9 runs, before fold 10)

Filled in 2026-09-30 16:35, from REPORT_seeds.md sections 1-3 (fold 9 only), before any fold-10 evaluation
of this round.

- Noise floor: resnet1d_wang seeds 0-4 fold-9 0.9357 +- 0.0009.
- Variants vs base 0.9365: SWA -0.0036, EMA -0.0038, label smoothing 0.05 +0.0018, crop 500/250 +0.0001.
  Only label smoothing exceeds the floor.
- **FINAL2_single = resnet1d_wang_crop_dsnorm_ls05** (seed 0, fold-9 0.9382).
- **FINAL2_ensemble = seeds 0-4 of resnet1d_wang, inception1d, xresnet1d50, base recipe** (15 models,
  fold-9 0.9403). Label smoothing is not added to the ensemble: the rule fixed the base recipe.
- Baselines: FINAL_single (resnet1d_wang_crop_dsnorm, scored in round 2, not re-run) and
  BASE_xresnet1d101 (xresnet1d101_crop_dsnorm, scored once now for the paired bootstrap).

Caveat recorded with the decision: label smoothing was picked as the best of 4 variants on one seed, with a
gain of about 2 seed stds. It is a weak selection, so its fold-10 gain should not be expected to hold fully.
