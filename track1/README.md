# PTB-XL Track 1

PTB-XL 100 Hz, 12 leads, 5 diagnostic superclasses (multilabel), official split
(folds 1-8 train / 9 val / 10 test). Records without a superclass are dropped (Strodthoff et al. 2021).

- `check_data.py` - fold counts, prevalence, dropped records, patient-leakage check
- `train.py --model M1|M2 --seed S` - M1 ResNet-SE CNN, M2 = M1 + 2-layer transformer
- `compare.py` - M1 vs M2 vs published references
- `calibrate.py` - Platt scaling, ECE, val-tuned thresholds, patient bootstrap CI (fold 9 fit only)
- `ensemble.py` - seeds 0-4 mean +- std and seed-ensemble
- `scripts/run_all.sh` - full resumable pipeline (gate for seeds 1-4: best seed-0 test mAUROC >= 0.92)

Data root: `$PTBXL_ROOT` (default `../data/ptb-xl`). Env: conda `medico` (`../environment.yml`).
Seed-0 report: `results/REPORT_s0.md`. Best checkpoints: `checkpoints/`.
