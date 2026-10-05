# benchmark: PTB-XL and Chapman-Shaoxing robustness study

PTB-XL 100 Hz, 12 leads, 5 diagnostic superclasses (multilabel), official split (folds 1-8 train, 9 val,
10 test). Records without a superclass are dropped (Strodthoff et al. 2021). Chapman-Shaoxing: four rhythm
classes, 80/10/10 split. Project overview and headline results: [../README.md](../README.md).

Run every module from this folder: `python -m <package>.<module>`.

## Code

| path | purpose |
|---|---|
| `loaders/ptbxl.py` | PTB-XL loading, band-pass, dataset-level standardization |
| `loaders/chapman.py` | Chapman-Shaoxing loading (same API) |
| `loaders/check_splits.py` | fold counts, prevalence, patient-leakage check |
| `noise/synthetic.py` | synthetic artifact families (baseline wander, EMG, motion burst, dropout, powerline, mixed) |
| `noise/nstdb.py` | real MIT-BIH NSTDB noise, split in time |
| `models/baselines.py` | first baselines M1 (ResNet-SE) and M2 (M1 + transformer) |
| `models/ptbxl_benchmark.py` | PTB-XL benchmark architectures (resnet1d_wang, xresnet1d, inception1d) |
| `models/gated.py` | Clean/Aug trunk and the Concat, Gate, DiffGate, SevGate variants on three trunks |
| `training/train.py` | training: `--aug`, `--aux`, crops, SWA/EMA, label smoothing, mixup, `--leads` |
| `training/decide_seeds.py`, `training/seed_ensemble.py` | seed-stage rule, seed summaries and ensembles |
| `evaluation/metrics.py` | macro-AUROC, ECE, patient bootstrap |
| `evaluation/robustness_eval.py` | corruption and NSTDB prediction grids, gate clamp sweep |
| `evaluation/calibration.py`, `evaluation/robustness_calibration.py` | Platt scaling and thresholds (fitted on fold 9) |
| `evaluation/difficulty_check.py` | severity rule of stage 1 |
| `evaluation/test_fold_eval.py`, `evaluation/fold10_log.py` | single fold-10 evaluations, each logged |
| `evaluation/compare_published.py` | comparison with published PTB-XL numbers |
| `external/single_lead_infer.py`, `external/cinc2017.py`, `external/butqdb.py` | single-lead models on CinC 2017 and BUT QDB |
| `reporting/grid_utils.py` | shared grid loading, seed averages, paired bootstraps |
| `reporting/*_report.py`, `reporting/rules3_analyses.py` | one report generator per stage |
| `scripts/` | resumable pipelines, one per stage |
| `tests/` | 85 tests: `pytest tests -q` |

## Results

- `results/ptbxl_baselines/` - PTB-XL benchmark runs, tables and selection rules (SELECTION2.md)
- `results/robustness/` - RULES, RULES2-4, FOLD10_LOG, stage reports (REPORT, nstdb, cinc2017, butqdb,
  revision_ablation, revision_analyses, rules4_report), figures; prediction grids are not in git
- `summaries/` - plain-language summaries of each round
- `logs/` - training and pipeline logs

Data root: `$PTBXL_ROOT` (default `../data/ptb-xl`); Chapman in `../data/chapman`. Env: conda `medico`
(`../environment.yml`). Checkpoints (`checkpoints/`) and caches (`cache/`) are not in git.

## Old names

The rules files, logs and summaries were written before the reorganization and use the old names.

| old | new |
|---|---|
| `track1/` | `benchmark/` |
| `phase1/` | `pilot/` |
| `results/track2/` | `results/robustness/` |
| `results/crop/` | `results/ptbxl_baselines/` |
| `reports/` | `summaries/` |
| `data.py`, `chapman.py`, `check_data.py` | `loaders/ptbxl.py`, `loaders/chapman.py`, `loaders/check_splits.py` |
| `corruptions.py`, `nstdb.py` | `noise/synthetic.py`, `noise/nstdb.py` |
| `models.py`, `strodthoff.py`, `track2_models.py` | `models/baselines.py`, `models/ptbxl_benchmark.py`, `models/gated.py` |
| `train.py`, `decide.py`, `ensemble.py` | `training/train.py`, `training/decide_seeds.py`, `training/seed_ensemble.py` |
| `metrics.py`, `track2_eval.py`, `final_eval.py`, `fold10_log.py` | `evaluation/metrics.py`, `evaluation/robustness_eval.py`, `evaluation/test_fold_eval.py`, `evaluation/fold10_log.py` |
| `calibrate.py`, `track2_calibration.py`, `track2_difficulty.py`, `compare.py` | `evaluation/calibration.py`, `evaluation/robustness_calibration.py`, `evaluation/difficulty_check.py`, `evaluation/compare_published.py` |
| `sl_infer.py`, `cinc2017.py`, `butqdb.py` | `external/single_lead_infer.py`, `external/cinc2017.py`, `external/butqdb.py` |
| `t2lib.py`, `crop_report.py`, `track2_report.py`, `secondary_severity.py` | `reporting/grid_utils.py`, `reporting/baseline_report.py`, `reporting/primary_verdict_report.py`, `reporting/secondary_severity_report.py` |
| `revision_ablation_report.py`, `revision_analyses.py` | `reporting/rules3_ablation_report.py`, `reporting/rules3_analyses.py` |
| `seed_report.py`, `seed_class_report.py`, `nstdb_report.py`, `fold10_report.py`, `cinc2017_report.py`, `butqdb_report.py`, `rules4_report.py` | same names in `reporting/` |
| `scripts/run_all.sh`, `run_seeds.sh`, `run_crops.sh` | `scripts/baselines_m1_m2.sh`, `ptbxl_seeds.sh`, `ptbxl_crops.sh` |
| `scripts/run_track2.sh`, `run_track2_eval.sh` | `scripts/rules1_train.sh`, `rules1_eval.sh` |
| `scripts/run_real.sh`, `run_singlelead.sh`, `run_external.sh` | `scripts/rules2_real_noise_train.sh`, `rules2_single_lead_train.sh`, `rules2_external_eval.sh` |
| `scripts/run_revision.sh`, `run_revision_eval.sh` | `scripts/rules3_ablation_train.sh`, `rules3_ablation_eval.sh` |
| `scripts/run_rules4.sh`, `run_rules4_eval.sh` | `scripts/rules4_train.sh`, `rules4_eval.sh` |

Commit hashes: the history was rewritten on 2026-10-05 (commit messages cleaned, paper sources removed);
commit dates are unchanged. The stage-1 rules commit is `4c77715`; hashes printed inside older training logs
refer to the commits before the rewrite.
