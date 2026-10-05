# Track 2 decision rules - written BEFORE any Track 2 run (2026-10-01)

Scope: PTB-XL 100 Hz, 12 leads, 5 superclasses, folds 1-8 train / 9 val / 10 test. Track 1 recipe
(resnet1d_wang, 2.5 s crops, sliding-window mean 250/125, band-pass 0.5-40 Hz, dataset-level
standardization fitted on train). All selection and all rules below use fold 9 only.

## A. Decision rules (Prompt 5, verbatim)

Candidate = E or F. Baselines = B0-aug, C, D (best baseline per group is the one
with the highest 3-seed probability-averaged macro-AUROC on that group).
Binding group: mixed (pooled over the chosen severities and modes).

GO if ALL hold for a candidate:
 1. Paired patient bootstrap (1000 resamples) difference vs best baseline on
    "mixed": 95% CI lower bound > 0 AND point estimate >= +0.005.
 2. On unseen_families the point estimate is >= +0.003 and CI lower bound > -0.002.
 3. r decreases with severity in a way large enough to matter: mean r at the
    lowest chosen SNR is at least 0.05 below mean r on clean records, AND
    Spearman(SNR, r) >= +0.3 (r rises with SNR).
 4. Clean-abnormal r >= clean-NORM r - 0.05.
 5. Clean macro-AUROC of the candidate is within 0.003 of B0-aug.
Otherwise NO-GO. With only 3 seeds, treat the paired bootstrap as the main
evidence and seed std as descriptive only. If the verdict is NO-GO, the
findings go into the benchmark-and-analysis paper.
Fold 10 is scored once, only for the winning candidate and the best baseline,
after the fold-9 verdict is written.

## B. How the rules are measured (fixed now, before any result)

- Predictions: every score uses the 3-seed probability average (seeds 0-2) unless it says per seed.
  r per record = mean of the per-window r over the sliding windows, then averaged over the 3 seeds.
- Group score = unweighted mean of macro-AUROC over the group's conditions:
  seen_families = baseline_wander, emg; unseen_families = motion_burst, dropout, powerline;
  mixed = the "mixed" family; all_corrupted = all of these. Each family is crossed with every chosen
  SNR and both modes (whole, burst).
- Paired bootstrap (rules 1-2): 1000 resamples of fold-9 patients (seed 0). In each resample, the same
  records are used for every condition and for both models; the group score of each model is
  recomputed and the difference is candidate - baseline. The point estimate is the difference on the
  full fold 9. The CI is the 2.5-97.5 percentile.
- Rule 3: "mean r at the lowest chosen SNR" pools all families and both modes at that SNR. "Mean r on
  clean records" is over all clean fold-9 records. Spearman is computed over all (record, condition)
  pairs of all_corrupted at the chosen SNRs, pairing the record's r with the condition's SNR in dB.
- Rule 4: clean fold-9 records with the NORM label vs records without it.
- Rule 5 is one-sided: candidate clean macro-AUROC >= B0-aug clean macro-AUROC - 0.003. A candidate
  that is better on clean data is not penalised.
- If both E and F pass, the winner is the one with the higher mixed group score. Fold 10 is then scored
  for the winner and for the best baseline on mixed. The fold-10 grid uses the same seeded corruptions,
  and each is reported with patient-bootstrap CIs and the paired difference. If the verdict is NO-GO,
  fold 10 is not scored.

## C. Severity rule (Prompt 2) and its reading

Use the lowest of {15, 6, 0, -6} dB at which the mean drop on "mixed" for the clean-trained model is
>= 0.03. If even -6 dB gives < 0.03, report that the backbone is robust to this benchmark and stop.

Reading, fixed now:
- "Lowest" means the mildest severity: scan 15 -> 6 -> 0 -> -6 dB and take the first SNR that meets the
  bar. "Even -6 dB" only makes sense with that reading.
- Mean drop = clean macro-AUROC minus mixed macro-AUROC at that SNR. It is computed per model (B0-clean
  seeds 0, 1, 2) and per mode (whole, burst), then averaged over these 6 values.
- Chosen severity set = the chosen SNR plus every harsher level of {15, 6, 0, -6}.
  Example: if 6 dB is chosen, the set is {6, 0, -6}.

## D. Training and model choices (fixed now)

- All runs early-stop on CLEAN fold-9 macro-AUROC (patience 10, max 50 epochs), exactly as in Track 1.
  Corrupted fold-9 data is never used for stopping or selection, except through the rules above.
- B0-clean = the existing resnet1d_wang_crop_dsnorm runs, seeds 0-2 (identical recipe, no retraining).
- Augmentation regime: corruption is added to the band-passed record (10 s) before standardization.
  The random 2.5 s crop is taken after that, then mixup. p = 0.5 per record. Training draws a random
  seen family, SNR ~ U[0, 20] dB and a random mode. Training noise comes from a numpy RNG seeded by
  (seed, epoch). The batch order, crops and mixup use the same torch generator as the clean runs.
- Shared trunk: resnet1d_wang. The shallow map is the output of stage 1: stem plus the first residual
  block, 3 of 7 conv layers, 128 channels at full time resolution. The deep map is the output of
  stage 3. F_local = Linear_128->d(GAP(shallow)) and F_ctx = Linear_128->d(GAP(deep)). A linear
  projection followed by GAP equals GAP followed by the same projection, so the cheaper order is used.
  d = 64.
- The gates in D, E and F are scalar per window: an MLP with 32 hidden units and one sigmoid output.
  P in E is a Linear d->d.
- Head for every variant: BN, Dropout 0.25, Linear(in, 128), ReLU, BN, Dropout 0.5, Linear(128, 5).
  This is the B0 head without its pooling; in = 2d for C and d otherwise.
- F: every sample gets a clean view and a corrupted view of the same record, with the same crop and the
  same mixup pairing and lambda. The corrupted view uses a seen family, U[0, 20] dB and a random mode.
  BCE is applied to the clean view or the corrupted view with p = 0.5 each, which gives the same input
  distribution as the augmentation regime.
  - L_consistency = MSE between clean and corrupted logits, only over samples whose corrupted view has
    SNR >= 15 dB. It is 0 if there are none.
  - L_severity = max(0, mean r(corrupted) - mean r(clean) + 0.05) over the batch.
  - Loss = BCE + 0.1 * L_consistency + 0.1 * L_severity.

## E. Chosen severity set (difficulty check, fold 9, 2026-10-01) and amendment

From difficulty.md, B0-clean seeds 0-2: the mean drop on mixed is 0.0003 at +15 dB, 0.0056 at +6 dB,
0.0228 at 0 dB and 0.0786 at -6 dB.
**Chosen severity set: {-6 dB}.** -6 dB is the only level that meets the 0.03 bar.

Amendment, written before any B0-aug or variant run: with one chosen level, "Spearman(SNR, r)" (rule 3)
and "r by SNR" have no SNR variation, so they are undefined. Therefore:
- Performance groups, rules 1, 2 and 5, and the fold-10 grid use the chosen set {-6 dB} only.
- Rule 3, first part: mean r at -6 dB (all families, both modes) vs mean r on clean records.
- Rule 3, second part, and the r diagnostics/figures use all four levels {15, 6, 0, -6} dB. Spearman is
  over all (record, condition) pairs of the corrupted conditions at those four levels.
- The fold-9 grid is therefore evaluated at all four levels. Performance-vs-SNR figures show all four;
  group scores use -6 dB only.
