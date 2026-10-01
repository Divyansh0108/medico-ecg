# Track 2, part 2 - rules for the follow-up analyses

These rules are written before the runs they govern. RULES.md (Track 2 decision rules) is unchanged and
its fold-9 verdict (NO-GO) stands; nothing below can change it. Every claim below is descriptive unless
a section says otherwise.

## 1. NSTDB (real recorded noise) - written 2026-10-01, before any NSTDB run

Data: MIT-BIH Noise Stress Test Database records bw (baseline wander), ma (muscle artifact) and em
(electrode motion), 2 channels each, 360 Hz, ~30 min (PhysioNet `nstdb` 1.0.0, SHA256 checked).

Preprocessing of the noise:
- Resample 360 -> 100 Hz with `scipy.signal.resample_poly(x, 5, 18)` (polyphase, default Kaiser window).
- Band-pass with `data.bandpass` (4th-order Butterworth 0.5-40 Hz, zero-phase): the same filter as the ECG.
- Time split of EACH record (in 100 Hz samples, T = record length): TRAIN = [0, floor(0.6 T)),
  gap = 10 s (1000 samples), EVAL = [floor(0.6 T) + 1000, T). The ranges are asserted disjoint. Eval
  noise excerpts come only from EVAL; training augmentation (optional runs) only from TRAIN.

Families and generation (fold-9 records, band-passed, not standardized; 12 leads):
- nstdb_bw, nstdb_ma, nstdb_em: one random excerpt (start uniform inside the pool, length = the noisy
  segment) from that record. For each of the 12 leads, one of the 2 noise channels is chosen at random
  (independently per lead). Each lead's noise is made zero-mean and unit power over the segment, then
  scaled so 10 log10(P_signal / P_noise) equals the target SNR, with P_signal = mean square of that
  lead over the whole 10 s record and P_noise = mean square of the added noise over the samples it
  covers (as in corruptions.py). Zero-power leads get no noise.
- nstdb_mixed: two distinct families of the three (chosen at random), each with its own excerpt and
  per-lead channel choice, each at unit power; summed, rescaled to unit power per lead, then scaled to
  the target SNR.
- Modes: whole (10 s) and burst (2-4 s at a random position; corruptions.segment).
- Order: band-pass(ECG) -> add noise -> dataset-level standardization with the train mean/std (the
  same order as the synthetic benchmark).
- RNG: numpy default_rng seeded by (20261001, family index, SNR + 100, mode index, ecg_id, 7), so every
  model sees identical noisy records and the noise does not depend on record order.

Evaluation (fold 9 only; fold 10 is not touched in this section):
- SNR {-6, 0} dB x {whole, burst} x 4 families = 16 conditions + clean.
- Models: B0-clean, B0-aug, C, D, E, F, E-clean, seeds 0-2 (existing checkpoints, no retraining).
  Scores use the 3-seed probability average (per-seed values are reported as mean +- std).
- Group scores, as in RULES.md B: unweighted mean of macro-AUROC over conditions.
  nstdb_<family> = that family at both SNRs and both modes (4 conditions); nstdb_all = all 16
  conditions. Also reported per SNR. Drop = clean score - group score.
- Paired patient bootstrap (1000 resamples, seed 0, the same code as Track 2): F, E, D, C vs B0-aug and
  B0-aug vs B0-clean, on nstdb_all and nstdb_mixed restricted to -6 dB (both modes; 8 and 2 conditions).
- r diagnostics (D, E, F, E-clean; 3-seed mean r per record): mean r per family x SNR x mode, and
  Spearman(SNR, r) over all (record, condition) pairs of the 16 conditions, also per family.
- Tests: measured SNR within 0.5 dB of the target; eval excerpts lie inside EVAL; TRAIN and EVAL are
  disjoint.

Rule (descriptive only): "F beats B0-aug on real noise" is claimed only if, on nstdb_all at -6 dB, the
paired CI lower bound is > 0 AND the difference is >= +0.005 (the same bar as Track 2 rule 1). The same
bar is reported, not claimed, for nstdb_mixed.

Optional (run only after everything above): B0-aug-real and F-real (seeds 0-2) - the B0-aug / F
recipes with the corrupted training copy drawn 50/50 from (a) the synthetic seen families exactly as
before and (b) NSTDB TRAIN noise (family uniform over nstdb_bw, nstdb_ma, nstdb_em, nstdb_mixed;
SNR ~ U[0, 20] dB; random mode). They are reported only as separate "trained on real noise" rows;
NSTDB families are not unseen for them.

## 2. Secondary severity tables - written 2026-10-01, before they are computed

These tables are descriptive and are NOT part of the Track 2 decision rule. They use the stored fold-9
predictions (results/track2/grid/main), with no training and no new inference.
- SNR 0 dB and +6 dB, each separately, both modes. Groups as in RULES.md B (seen_families,
  unseen_families, mixed, all_corrupted) at that SNR only.
- Models: B0-clean, B0-aug, C, D, E, F, E-clean. 3-seed probability average, per-seed mean +- std,
  drop vs clean, and the paired patient bootstrap (1000, seed 0) of every model vs B0-aug.
- r (D, E, F, E-clean): mean r on clean vs mean r at that SNR (all families, both modes), and
  Spearman(SNR, r) over the 4 levels (as in the RULES.md amendment; same for both tables).
- A table per condition (family x mode) of the drop vs clean, to show which conditions cause drops.
- Figure: group score vs SNR (clean, +15, +6, 0, -6) for B0-clean, B0-aug, D, E, F.

## 3. CinC 2017 (single-lead transfer) - written 2026-10-01, before any single-lead run

A. Single-lead PTB-XL models. resnet1d_wang trunk with 1 input channel = PTB-XL lead I (index 0).
   The same recipe as Track 2: band-pass, dataset-level mean/std fitted on train lead I only, 2.5 s
   random crops, mixup 0.4, cosine schedule, early stopping on CLEAN fold-9 macro-AUROC (patience 10,
   max 50). Variants B0-clean (no aug), B0-aug, D, E, F (aux losses, unchanged weights) and E-clean,
   seeds 0-2, i.e. 18 runs (train.py --leads 0). Corruption augmentation is applied to the single lead
   exactly as to each lead before. Reported: clean fold-9 macro-AUROC (3-seed average and per seed).
   Fold 10 is not scored for these models.
B. Data. CinC 2017 training set, 8,528 recordings, 300 Hz. Labels from REFERENCE-v3.csv (the label
   file shipped with the downloaded set; v3 is the final relabelled version): N, A, O, ~.
   - Resample 300 -> 100 Hz (resample_poly(x, 1, 3)), band-pass with data.bandpass.
   - Main: standardize with CinC 2017's own pooled mean/std over all samples of all recordings (labels
     not used). Sensitivity: per-record z-score.
   - Windows: 10 s (1000 samples), starts 0, 500, 1000, ... while the window fits, plus one final
     window ending at the last sample if the tail is not covered. Records shorter than 10 s are
     reflect-padded (numpy "reflect", repeated if needed) to 1000 samples (one window).
   - Each window is scored like a PTB-XL record (2.5 s crops, stride 1.25 s, mean over crops).
     Per-recording r = mean over windows (also reported: min over windows). r is averaged over seeds 0-2.
C. Noisy detection. Target: "~" vs rest. The noise score of r is -r (low r = noisy; F was trained so r
   falls with corruption). AUROC with a recording bootstrap 95% CI (1000 resamples, seed 0), for D, E,
   F, E-clean, with mean-r and min-r.
   Heuristics (per recording, on the 100 Hz band-passed signal):
   - H1 = power(20-40 Hz) / power(0.5-20 Hz), Welch PSD (nperseg 256), high = noisy.
   - H2 = max |x| of the per-record z-scored signal, high = noisy.
   Best heuristic = the one with the higher full-data AUROC. Paired recording bootstrap (1000, seed 0)
   of AUC(r) - AUC(best heuristic).
   Claim "r separates Noisy" (per model) only if the AUC CI lower bound > 0.5 AND the paired CI vs the
   best heuristic lies entirely above 0. The main analysis is mean-r with pooled standardization.
D. Frozen-representation probe. Features = the vector that enters the classifier, averaged over
   crops and windows of a recording: for B0 (resnet1d_wang) the 256-d concat pooling (max, mean) of the
   head; for F the 64-d fused feature r F_local + (1 - r) F_ctx. Models B0-clean, B0-aug, F, each seed
   separately. Probe: StandardScaler + LogisticRegression (multinomial, C = 1.0, max_iter 5000, no
   class weights, no tuning), RepeatedStratifiedKFold(5 folds, 3 repeats, random_state 0), by recording.
   Metrics on each test fold: challenge F1 = mean(F1_N, F1_A, F1_O) and AF-vs-rest AUROC (probability
   of A). Reported: mean +- std over the 15 folds for each seed, and the mean over seeds.
Notes for the report: no patient IDs, so folds are record-level; the hidden test set is not
available, so scores are not comparable to the leaderboard; nothing is retrained or tuned on CinC 2017.
