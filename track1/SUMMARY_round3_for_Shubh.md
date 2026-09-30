# PTB-XL Track 1 - round 3: seeds, SWA/EMA, label smoothing, crop 500, ensemble (2026-09-30)

12 leads, 100 Hz, official folds (1-8 train / 9 val / 10 test). Recipe: 2.5 s random crops, sliding-window
mean (250/125), dataset-level norm fitted on train, band-pass on. All selection on fold 9. The selection
rule was written in `results/crop/SELECTION2.md` before the runs, and the final configurations were fixed
there before fold 10 was opened. Round 2 is in `SUMMARY_for_Shubh.md`.

## Headline (fold 10, each entry evaluated once)

| entry | fold 9 | fold 10 | 95% CI (patient bootstrap) | published |
|---|---|---|---|---|
| Ensemble: 15 models (seeds 0-4 of resnet1d_wang, inception1d, xresnet1d50) | 0.9403 | **0.9363** | 0.9291-0.9428 | ensemble .934 |
| Single: resnet1d_wang + label smoothing 0.05 (seed 0) | 0.9382 | **0.9335** | 0.9259-0.9401 | resnet1d_wang .930 |
| resnet1d_wang reproduced in this pipeline (seed 0) | 0.9365 | 0.9332 | 0.9255-0.9399 | .930 |
| xresnet1d101 reproduced in this pipeline (seed 0) | 0.9311 | 0.9309 | 0.9233-0.9378 | .928 |

## 1. Seeds 0-4 (fold 9)

| model | mean +- std | 5-seed probability average |
|---|---|---|
| resnet1d_wang | 0.9357 +- 0.0009 | 0.9394 |
| inception1d | 0.9337 +- 0.0008 | 0.9389 |
| xresnet1d50 | 0.9325 +- 0.0009 | 0.9375 |

## 2-3. One change at a time, resnet1d_wang seed 0 (fold 9, base 0.9365)

| change | fold 9 | vs base |
|---|---|---|
| SWA over epochs 41-50 | 0.9329 | -0.0036 |
| EMA (decay 0.999) over epochs 41-50 | 0.9327 | -0.0038 |
| label smoothing 0.05 | 0.9382 | +0.0018 |
| crop 500, stride 250 | 0.9366 | +0.0001 |

SWA and EMA runs train all 50 epochs without early stopping; BatchNorm statistics are recomputed on train
after averaging. Rule: a change is adopted only if its gain is larger than the seed std (0.0009), so only
label smoothing was adopted, for the single model.

## 4. Ensemble

Pre-specified rule: mean of probabilities of all seeds of resnet1d_wang, inception1d and xresnet1d50
(15 models), with no subset search and no weights. Fold 9: 0.9403. Fold 10: 0.9363.

## 5. Paired patient bootstrap on fold 10 (1000 resamples), ours - reproduced baseline

| ours | baseline | difference | 95% CI | P(diff <= 0) |
|---|---|---|---|---|
| ensemble | resnet1d_wang | +0.0031 | +0.0012 to +0.0051 | <0.001 |
| ensemble | xresnet1d101 | +0.0054 | +0.0031 to +0.0076 | <0.001 |
| single | resnet1d_wang | +0.0002 | -0.0015 to +0.0018 | 0.40 |
| single | xresnet1d101 | +0.0026 | -0.0002 to +0.0054 | 0.03 |

## Caveats

- Fold-10 scores of the ensemble members over seeds 0-4 (a by-product of scoring the ensemble, not used for
  any choice): resnet1d_wang 0.9315 +- 0.0010, inception1d 0.9295 +- 0.0013, xresnet1d50 0.9280 +- 0.0014.
  Seed 0 (0.9332) was the best resnet1d_wang seed, so 0.9315 is the fair single-model number.
- Label smoothing gave +0.0018 on fold 9 but +0.0002 on fold 10. It was the best of 4 variants on one seed,
  so the fold-9 gain was mostly selection noise.
- The 15-model ensemble (0.9363) is not better than the 5-model ensemble of round 2 (0.9364).
- The ensemble is significantly better than both reproduced single models, but its margin over the
  published ensemble (.934) is +0.002 and its CI includes .934. This supports "matches or slightly exceeds
  published", not a clear improvement.
- Fold 10 has now been scored for 8 entries in total (5 in rounds 1-2, 3 in this round). (Corrected 2026-10-01: an earlier version said 7.)
- Train times in the round-3 JSON files are inflated: three training queues shared the GPU.

## Files

- `results/crop/REPORT_seeds.md` - all tables of this round; `results/crop/SELECTION2.md` - rule and decision
- `results/crop/FINAL2_single.json`, `FINAL2_ensemble.json`, `BASE_xresnet1d101.json` - fold-10 results
- `results/crop/*.json`, `results/crop/probs/` - per-run metrics, histories, probabilities
- `train.py` (new: `--avg swa|ema --avg-epochs --ema-decay --label-smooth`), `seed_report.py`,
  `metrics.py` (`paired_patient_bootstrap`), `scripts/run_seeds.sh` (all runs of this round)
- `checkpoints/` - weights of every run; `logs/seeds_*.log`, `logs/final2.log` - logs
