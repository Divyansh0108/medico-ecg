# track1 - PTB-XL benchmark, corruption benchmark and reliability-gate experiments

PTB-XL 100 Hz, 12 leads, 5 diagnostic superclasses (multilabel), official split
(folds 1-8 train / 9 val / 10 test). Records without a superclass are dropped (Strodthoff et al. 2021).
Project overview and headline results: [../README.md](../README.md).

## Code

| file | purpose |
|---|---|
| `data.py`, `check_data.py` | loading, dataset-level standardization, fold counts, patient-leakage check |
| `models.py`, `strodthoff.py` | M1/M2 and the published-baseline architectures (resnet1d_wang, inception1d, xresnet1d) |
| `track2_models.py` | B0, C, D, E, F variants (trunk split after stage 1, params within 1% of B0) |
| `train.py` | training: `--aug`, `--aux`, crops, SWA/EMA, label smoothing, mixup |
| `corruptions.py`, `nstdb.py` | synthetic corruption families (53 tests) and real NSTDB noise |
| `calibrate.py`, `track2_calibration.py`, `metrics.py` | Platt scaling, ECE, thresholds, patient bootstrap |
| `ensemble.py`, `final_eval.py`, `decide.py` | seed ensembles, fold-10 evaluation, decision rules |
| `track2_eval.py`, `track2_difficulty.py`, `track2_report.py` | Track 2 grid evaluation and reports |
| `cinc2017.py`, `butqdb.py` (+ `*_report.py`) | external quality-label checks |
| `fold10_log.py`, `fold10_report.py` | logged, single-pass test-fold reporting |
| `scripts/` | resumable pipelines, one per stage |
| `tests/` | 85 tests: `pytest tests -q` |

## Results and reports

- `reports/` - plain-language summaries: `round2_summary.md`, `round3_summary.md`, `track2_summary.md`
- `results/REPORT_s0.md` - M1 vs M2 seed 0
- `results/crop/` - round 2-3 JSON, tables, selection rules
- `results/track2/` - REPORT, RULES, RULES2, FOLD10_LOG, nstdb / cinc2017 / butqdb reports, figures
  (prediction grids, ~120 MB, are not in git)
- `logs/` - training and pipeline logs

Data root: `$PTBXL_ROOT` (default `../data/ptb-xl`). Env: conda `medico` (`../environment.yml`).
Checkpoints and caches are not in git.
