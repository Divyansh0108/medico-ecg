# PTB-XL Track 1 - crops, backbones, ablations (2026-09-30, seed 0)

12 leads, 100 Hz, official folds (1-8 train / 9 val / 10 test), seed 0. All selection on fold 9.
Final configurations fixed in `results/crop/SELECTION.md` before fold 10 was opened.

## Headline (fold 10)

| model | fold 9 | fold 10 | 95% CI (patient bootstrap) | published |
|---|---|---|---|---|
| Ensemble: 5 models, crops + dataset norm | 0.9388 | **0.9364** | 0.929-0.943 | ensemble .934 |
| resnet1d_wang, crops + dataset norm | 0.9365 | **0.9332** | 0.926-0.940 | resnet1d_wang .930 |
| M1, crops (item 1) | 0.9144 | 0.9129 | 0.904-0.920 | - |
| M1, original (no crops) | 0.8994 | 0.8967 | - | - |

Published (Strodthoff et al. 2021): xresnet1d101 .928, resnet1d_wang .930, inception1d .921, ensemble .934.

## What was done
1. Random 2.5 s crops (250 samples) for training, and sliding windows (250, stride 125) averaged per
   record for evaluation. Applied to M1: +0.016 on fold 10.
2. Backbones re-implemented from the Strodthoff repo: resnet1d_wang (0.47M params), xresnet1d50 (0.95M),
   xresnet1d101 (1.87M), inception1d (0.51M). Same crops, loss, folds and seed. Params and train time are
   in REPORT.md.
3. M1 ablations, one at a time (fold 9): dataset-level standardization **+0.018** (mostly HYP: .837 -> .913,
   because per-record z-score removes QRS amplitude, a hypertrophy criterion); band-pass off 0.000;
   mixup off -0.002; one-cycle with peak 1e-2 -0.017 (peak too high).
4. Extra step, not in the original ask: the 4 backbones were retrained with dataset-level standardization,
   which gave +0.016-0.019 for each of them on fold 9.
5. Final: an ensemble of all 5 dataset-norm models, a pre-specified rule rather than the best fold-9 subset
   (the top subsets differ by <0.0005). The best single model on fold 9 is also reported.

## Caveats
- Single seed (0). Margins over published (+0.002 ensemble, +0.003 single) are inside the CIs, so
  seeds 1-4 (mean +- std) are needed before claiming SOTA. Published numbers are also means over runs.
- Fold 10 was scored for 3 entries: M1-crop (item 1 asked for it), the final single model and the final
  ensemble. The per-member fold-10 scores printed while scoring the ensemble were not used for any choice.
- Test aggregation uses mean over windows, as requested. Strodthoff's code uses max; on fold 9, max
  scored 0.001-0.003 lower for every run.

## Files
- `results/crop/REPORT.md` - full tables (all runs, fold-9 ensembles, fold 10 vs published)
- `results/crop/SELECTION.md` - final choice recorded before fold 10
- `results/crop/*.json`, `results/crop/probs/` - per-run metrics, histories, probabilities
- `strodthoff.py` (backbones), `train.py` (`--crop --norm --no-bandpass --mixup --sched`),
  `final_eval.py` (one-shot fold-10), `crop_report.py`, `scripts/run_crops.sh` (full pipeline)
- `checkpoints/` - best weights for every run; `logs/crops*.log` - training logs
