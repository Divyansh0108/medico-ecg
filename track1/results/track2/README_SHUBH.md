# Track 2 follow-ups: summary

PTB-XL, 100 Hz. Folds 1-8 = train, fold 9 = validation, fold 10 = test. Every rule was written into `RULES2.md` and committed before its run. All results below are descriptive and do not change the Track 2 verdict (fold-9 NO-GO: on mixed noise F was only +0.002 vs B0-aug, short of the +0.005 bar). Commit: 1bfdb27.

## Headlines

1. **NSTDB real noise, fold 9** (`nstdb.md`)
   - F - B0-aug on nstdb_all at -6 dB = -0.0096 [-0.0117, -0.0074], so the claim "F beats B0-aug on real noise" is **not supported**. D, E and C are also below B0-aug.
   - Synthetic augmentation helps on real noise: B0-aug - B0-clean = +0.025.
   - r falls as real noise increases.
   - **Trained on real noise** (separate rows): B0-aug-real and F-real both gain +0.025 over B0-aug. F-real - B0-aug-real = -0.0006 [-0.0021, +0.0009], so the gate adds nothing once real noise is in the training data.

2. **Secondary severity at 0 and +6 dB, fold 9** (`secondary_severity.md`): every model is within about 0.002 of B0-aug.

3. **Per-seed results and r by class** (`seed_and_class.md`)
   - On mixed@-6, F's +0.002 over B0-aug comes from averaging the 3 seeds. The mean of the per-seed differences is about 0.
   - r is lowest on NORM-only records.

4. **Fold 10, a single reporting pass** (`fold10_report.md`; every scoring is logged in `FOLD10_LOG.md`, which includes the disclosure)
   - Clean: all models score 0.9347.
   - F - B0-aug: all_corrupted@-6 +0.0023 [+0.0006, +0.0039]; mixed@-6 +0.0025 [+0.0003, +0.0044]; NSTDB all@-6 -0.0081 [-0.0100, -0.0062].
   - This is consistent with fold 9: there is a small gain on synthetic noise, below the +0.005 bar, and a loss on real noise.

5. **Single-lead models (lead I)**: clean fold-9 macro-AUROC is about 0.84 for every variant, vs about 0.94 for 12-lead. Several runs reached the 50-epoch cap.

6. **CinC 2017** (`cinc2017.md`)
   - Main analysis (pooled standardization, mean r): **"r separates Noisy" is not supported for any model.**
     - D, E and E-clean have AUROC 0.35-0.40, i.e. r is *higher* on noisy recordings.
     - F has AUROC 0.54 [0.49, 0.58].
   - Sensitivity analysis with per-record z-scoring: AUROC 0.70-0.77, with F best at 0.765 and +0.31 above the best heuristic. This was not the pre-registered main analysis.
   - Frozen-feature probe (F1 over N/A/O; AF AUROC): B0-aug 0.674 / 0.949, B0-clean 0.666 / 0.945, F 0.565 / 0.913.

7. **BUT QDB** (`butqdb.md`)
   - r is **higher in class 2 than in class 1** for every gated model. The CI excludes 0, so this is the wrong direction.
   - r rises with accelerometer motion (Spearman about +0.4).
   - Class 2 windows have about 10x the motion of class 1 windows.
   - The ordering class 1 > 2 > 3 holds in at most 1 of the 6 eligible subjects.
   - H2 (max |z|) also runs in the "wrong" direction in a consistent way.
   - Caveats:
     - The protocol in the Scientific Data descriptor was not accessible, so the analysis uses the assumptions A1-A8 listed in `RULES2.md`.
     - Class 3 is dominated by subject 105.
     - Class 2 may be mostly baseline drift, which the 0.5-40 Hz band-pass removes.

**Bottom line:** the gate r does not work as a quality signal outside the synthetic benchmark. Under real noise and on external annotated quality data it is flat or inverted. The only gain that holds up comes from noise augmentation itself, and adding real noise to it helps much more than the gate does.

## Contents

- `results/track2/*.md, *.json`: reports and the numbers behind them. `RULES2.md` holds the pre-registered rules; `FOLD10_LOG.md` logs every test-fold access.
- `results/track2/figures/`: figures.
- `results/track2/runs_sl/`: per-run JSON for the single-lead models.
- `results/track2/butqdb/windows_log.csv`: kept and dropped windows per BUT QDB record.
- `code/`: the Python code, scripts and tests.
- `logs/`: training and pipeline logs.

Not included: checkpoints, cached arrays (`*.npz`, `*.pt`) and the datasets.
