# Track 2, part 3 - revision ablations and analyses (written 2026-10-03, before any run below)

Motivation: reviewer questions on the paper draft (sensitivity of the severity loss, gate capacity,
time-resolved gating, augmentation mix, per-class and calibration behaviour under noise, diagnostic
signal in r). RULES.md and RULES2.md are unchanged; the fold-9 verdict (NO-GO) and the fold-10 pass
stand. Nothing below can change them. **Everything below is descriptive, fold 9 only. Fold 10 is not
touched.** No new GO/NO-GO decision is made.

## 1. Ablation runs (new training; recipe of train.py with --norm dataset --crop 250, seeds 0-2)

| tag prefix | paper name | change from the reference variant |
|---|---|---|
| Fm002 | SevGate, margin 0.02 | t2_E --aux --sev-margin 0.02 |
| Fm010 | SevGate, margin 0.10 | t2_E --aux --sev-margin 0.10 |
| Fws03 | SevGate, w_sev 0.3 | t2_E --aux --w-sev 0.3 |
| Fwc0 | SevGate, no consistency | t2_E --aux --w-cons 0 |
| Fh128 | SevGate, 128 gate units | t2_E_h128 --aux |
| T | TimeGate | t2_T --aug (time-resolved DiffGate, same parameter count as DiffGate) |
| TF | TimeSevGate | t2_T --aux (time-resolved SevGate; severity loss on mean_t r(t)) |
| B0augall | Aug-all | resnet1d_wang --aug --aug-families all |
| Fall | SevGate-all | t2_E --aux --aug-families all |

Gate-input ablation (with vs without the disagreement term) already exists: Gate (D) vs DiffGate (E).
"all" = synthetic augmentation drawn uniformly from all five families (motion_burst, dropout and
powerline are then seen; their rows are not "unseen" and are reported separately).

## 2. Evaluation of the ablations (fold 9)

- Synthetic grid: clean + 6 families x {15, 6, 0, -6} dB x {whole, burst} (as grid/main), written to
  grid/ablation. Real noise: NSTDB eval range, {0, -6} dB (as grid/nstdb), written to grid/ablation_nstdb.
- Scores: 3-seed probability average; groups as in RULES.md B at -6 dB (seen, unseen, mixed,
  all_corrupted) and nstdb_all at -6 dB. Per-seed mean +- s.d. also reported.
- Paired patient bootstrap (1000 resamples, seed 0, t2lib) of every ablation vs Aug (B0-aug) and vs the
  reference SevGate (F) on mixed@-6, unseen@-6 and nstdb_all@-6; Aug-all and SevGate-all also vs each other.
- r diagnostics as in RULES.md rule 3 (r drop clean -> -6 dB, Spearman(SNR, r) over 4 levels) and rule 4.
- Reading: an ablation is called "a practically relevant gain over Aug" only with the RULES.md rule-1
  bar (mixed@-6 CI lower bound > 0 and point estimate >= +0.005). This is reported, not used to pick a
  model; with 9 ablations, multiplicity is acknowledged in the text.

## 3. Analyses of existing stored predictions (grid/main, grid/nstdb; no new training)

a. Per-class AUROC (NORM, MI, STTC, CD, HYP) under noise: clean, mixed@-6, seen@-6, unseen@-6,
   nstdb_all@-6 for every model (3-seed average); paired bootstrap F - B0-aug per class on mixed@-6
   and nstdb_all@-6.
b. Calibration under noise: ECE (15 equal-width bins, mean over classes) and Brier score (mean over
   classes) per SNR for whole-record mixed noise and NSTDB mixed, raw and after per-class Platt scaling.
   Platt is fitted on CLEAN fold-9 records only, with 2-fold patient-level cross-fitting (fit on one
   half of the patients, evaluate on the other half, swap; split seed 0), so no record is scored by a
   calibrator fitted on it.
c. Diagnostic signal in r (clean fold 9; D, E, F, E-clean; 3-seed mean r): AUROC of r for any-abnormal
   (non-NORM) vs NORM; per superclass, AUROC of r alone; 5-fold patient-grouped logistic regression of
   each label from r (reported as AUROC). Residual r-noise relation after removing the diagnosis:
   Spearman(SNR, delta r) with delta r = r(condition) - r(clean, same record), which removes any
   record-level (diagnosis-dependent) baseline; also within NORM-only and non-NORM strata.
d. Moderate severities: group scores at 0 and +6 dB already in secondary_severity.md, moved to an
   appendix table.
