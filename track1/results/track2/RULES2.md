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
B. Data. CinC 2017 training set, 8,528 recordings, 300 Hz. Labels: training2017/REFERENCE.csv (as
   specified; N 5050, A 738, O 2456, ~ 284). Sensitivity: REFERENCE-v3.csv (the later relabelled
   version; N 5076, A 758, O 2415, ~ 279; 148 records differ).
   [Amended 2026-10-01, before any CinC 2017 run: the first version said REFERENCE-v3.csv and wrongly
   called it the file shipped with the set. The zip ships REFERENCE.csv, which differs from v3.]
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

## 4. Freeze and the single fold-10 reporting pass - written 2026-10-01, before any fold-10 scoring in this part

From this point no model or hyperparameter changes are made to the 12-lead Track 2 models B0-clean,
B0-aug, C, D, E, F (and E-clean). Their checkpoints (seeds 0-2) are final. The runs still in progress
use recipes fixed earlier in this file: the single-lead models (section 3, fixed before they started)
and the optional real-noise models (section 1). They are never scored on fold 10 and nothing is tuned
on any result.

ONE fold-10 reporting pass. It is descriptive, not selection, and nothing is chosen from it:
- Models: B0-clean, B0-aug, C, D, E, F; 3-seed probability average and each seed.
- Conditions on fold 10 (the same seeded generators as fold 9, keyed by ecg_id): clean; the synthetic
  families at -6 dB and 0 dB, both modes; the NSTDB families (EVAL noise) at 0 and -6 dB, both modes.
- Groups: clean; synthetic seen_families, unseen_families, mixed and all_corrupted, each at -6 dB and at
  0 dB; nstdb_bw, nstdb_ma, nstdb_em, nstdb_mixed and nstdb_all (0 and -6 dB pooled), plus nstdb_all
  and nstdb_mixed at -6 dB.
- Patient bootstrap 95% CIs (1000 resamples, seed 0) of every group score, and paired differences vs
  B0-aug on the same resamples.
- Every model and every 3-seed average scored on fold 10 is logged in FOLD10_LOG.md. track2_eval.py
  refuses to score a tag twice under the same grid name.

## 5. BUT QDB (real ambulatory quality labels) - written 2026-10-01, before any BUT QDB run

Source and protocol. PhysioNet butqdb 1.0.0 (the only version published; all 95 files SHA256-checked).
The README (PhysioNet page) was read. The Scientific Data descriptor (Smital et al. 2026,
doi:10.1038/s41597-026-07905-w) says it provides "a standardized evaluation protocol and open-source
code", but only its abstract was accessible: the full text is behind a login (Europe PMC: "subscription
required"), and no official code repository could be found (GitHub search: only third-party repos).
The descriptor's protocol is therefore NOT followed here, because it could not be read. Everything below
follows the PhysioNet README and is listed as an assumption. If the descriptor becomes available, this
section is re-checked against it before any claim is made.

Verified from the files (printed by butqdb.py): 18 recordings, 15 subjects (subject = first 3 digits of
the record name; 100 has 2 recordings, 103 has 3). ECG 1 channel at 1000 Hz, ACC 3 channels (x, y, z) at
100 Hz. Consensus-annotated time: class 1 51.2 h, class 2 33.1 h, class 3 15.1 h (class 3 is concentrated
in record 105001).

Classes (README wording, verified): 1 = P, QRS and T clearly visible, onsets and offsets reliable;
2 = noise increased, significant points unreliable, QRS clearly visible and reliably detectable;
3 = QRS not reliably detectable, signal unsuitable for analysis. 0 = not annotated.

Assumptions:
A1. Labels = the consensus columns (10-12). The 3 annotators' own columns are not used.
A2. Annotation sample indices are 1-based and inclusive at the ECG rate, 1000 Hz (as in ann_reader.m;
    the last end equals the record length). Class 0 spans are excluded.
A3. Windows: a fixed non-overlapping 10 s grid from the start of each recording. A window is kept only if
    all of its 10,000 ECG samples carry one consensus class in {1, 2, 3}. Grid windows that touch
    annotated samples but span two classes, or include class-0 samples, are counted as dropped. No
    official split exists in the README, so there is no split: everything is evaluation, nothing is
    fitted.
A4. ECG: resample_poly(x, 1, 10) over the whole recording, then data.bandpass. Main standardization:
    BUT QDB pooled mean/std over all samples of the kept windows. Sensitivity: each 10 s window
    z-scored on its own (PTB-XL's per-record z-score is also per 10 s record).
A5. The BUT QDB lead (Bittium Faros 180, chest-worn) is not PTB-XL lead I; the lead-I models are applied
    as they are. This is a domain shift and is stated as a caveat.
A6. ACC: vector magnitude sqrt(x^2 + y^2 + z^2) in the units of the header, high-pass 0.5 Hz (4th-order
    Butterworth, zero-phase) over the whole recording, RMS over the window's 1000 ACC samples
    (ACC index = ECG index / 10).
A7. Heuristics H1 and H2 as in section 3, computed per 10 s window on the band-passed 100 Hz signal.
A8. Models: single-lead D, E, F, E-clean, seeds 0-2, 3-seed average of the per-window r (window scored as
    in section 3). No retraining, nothing tuned on BUT QDB.

Analyses (subject-level bootstrap: 1000 resamples of the 15 subjects with replacement, seed 0; a resampled
subject brings all its windows from all its recordings):
a. Mean r per class (pooled windows) and differences between classes (1-2, 2-3, 1-3); Spearman(class, r)
   over pooled windows; pairwise AUROC 1v2, 2v3, 1v3 within each subject with >= 5 windows in both
   classes, then averaged over those subjects. Orientation: the score is -r (for H1 and H2 the score is
   the value), so AUROC > 0.5 means worse quality gets a lower r.
b. Number of subjects (among those with >= 5 windows in each of the 3 classes) whose mean r follows
   class 1 > 2 > 3 (H1, H2: 1 < 2 < 3).
c. Spearman(motion RMS, r): pooled and within each class.
d. The same quantities for H1 and H2.
Claims: with 15 subjects, a finding is claimed only if its subject-level 95% CI excludes 0 (for AUROC,
the CI of AUROC - 0.5). Caveat to state: class 2 may be dominated by baseline-drift noise, which the
0.5-40 Hz band-pass partly removes before any model or heuristic sees it.
