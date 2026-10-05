# Track 2, part 4 - robustness of the conclusion (written 2026-10-04, before any run below)

Motivation: test whether the RULES.md conclusion (reliability gating adds nothing practically relevant
beyond noise augmentation) holds across architectures, a second dataset, more seeds, gate settings and
stricter statistics. RULES.md-RULES3.md are unchanged and nothing below changes their verdicts. PTB-XL
fold 10 is not touched. Every recipe is train.py --norm dataset --crop 250 (as RULES.md/RULES2.md).

## 1. Settings and runs

| setting | trunk | data, eval split | variants (seeds) |
|---|---|---|---|
| S1 wang | resnet1d_wang | PTB-XL, fold 9 | existing Clean/Aug/Gate/DiffGate/SevGate seeds 0-2 (+ Clean 3-4 exist); NEW Aug, Gate, DiffGate, SevGate seeds 3-4 (8 runs) |
| S2 xresnet | xresnet1d50 | PTB-XL, fold 9 | Clean exists (seeds 0-2); NEW Aug, DiffGate, SevGate, DiffGate-clean, Aug-real, SevGate-real, seeds 0-2 (18 runs) |
| S3 inception | inception1d | PTB-XL, fold 9 | as S2 (18 runs) |
| S4 chapman | resnet1d_wang | Chapman-Shaoxing, held-out test | Clean, Aug, DiffGate, SevGate, DiffGate-clean, Aug-real, SevGate-real, seeds 0-2 (21 runs) |

Gated trunks for S2/S3 follow RULES.md D: the trunk is split into a shallow and a deep part, F_local and
F_ctx are linear projections (d = 64) of the global average of each part, the DiffGate gate MLP (32 hidden
units) sees [F_local, F_ctx, |F_local - P F_ctx|], and the head is the Track 2 head.
- xresnet1d50: shallow = stem + max-pool + stage 1 (256 ch); deep = stages 2-4 (256 ch).
- inception1d: shallow = inception blocks 1-3 + shortcut 1 (128 ch); deep = blocks 4-6 + shortcut 2 (128 ch).
Training flags as in RULES.md/RULES2.md (Aug --aug; DiffGate --aug; SevGate --aux; DiffGate-clean no aug;
Aug-real --aug-real; SevGate-real --aux --aug-real).

Chapman-Shaoxing (PhysioNet/CinC 2021 copy, 12 leads, 500 Hz, 10 s). The four rhythm classes of Zheng et
al. 2020, from the SNOMED codes in the headers:
SB {426177001}; AFIB {164889003 (AF), 164890007 (AFL)};
GSVT {427084000 (ST), 426761007 (SVT), 713422000 (AT), 233896004 (AVNRT), 233897008 (AVRT)};
SR {426783006 (SR), 427393009 (SA)}.
Records whose codes map to no class or to more than one class are dropped, and so are records with a
non-finite sample, a length other than 5000, or a flat lead (std < 1e-6 mV). The signal is
resampled to 100 Hz (scipy resample_poly 1/5, 1000 samples); the processing after that is the same as
PTB-XL (0.5-40 Hz band-pass, noise added before dataset-level standardization with the train mean/std).
One record per patient. Split stratified by class 80/10/10 (train/val/test), seed 20261004. Labels are
one-hot over the four classes, the loss is the same BCE, and the metric is the one-vs-rest macro-AUROC.
Early stopping uses val; every robustness number uses test (never used for selection).

## 2. Evaluation grids

Every setting: clean + 6 families x {15, 6, 0, -6} dB x {whole, burst} (as grid/main), and NSTDB eval
noise at {0, -6} dB (as grid/nstdb). Groups at -6 dB, both modes: seen, unseen, mixed, all_corrupted,
nstdb_all. Scores use the probability average over seeds (S1: seeds 0-4 for Clean/Aug/Gate/DiffGate/
SevGate; the 3-seed numbers are also reported).

## 3. Confirmatory comparisons (Holm-corrected)

Primary endpoint: macro-AUROC on mixed@-6. Family of 8 one-sided tests: {DiffGate, SevGate} - Aug in
S1-S4, H1: difference > 0. p = fraction of paired patient-bootstrap resamples (1000, seed 0, t2lib) with
difference <= 0. Holm step-down at familywise alpha = 0.05. A gate "helps" only if the Holm test rejects
AND the point estimate is >= +0.005 (the RULES.md practical bar).
Equivalence (TOST): the same 8 differences are called equivalent to zero at margin +-0.005 if the 90%
bootstrap CI lies inside (-0.005, +0.005). They are called inconclusive if neither test passes.
Secondary endpoints, same tests reported without being part of the family: unseen@-6, nstdb_all@-6,
clean; SevGate-real - Aug-real on nstdb_all@-6 (S2-S4).

## 4. Effect sizes and pooling

- Delta AUROC with 95% CI (paired bootstrap).
- Cohen's d_z across seeds: seed-matched per-seed differences (seed k gate vs seed k Aug) of the group
  score, mean / s.d.
- Share of the augmentation gain: (gate - Aug) / (Aug - Clean), on the point estimates.
- Random-effects meta-analysis (DerSimonian-Laird) of gate - Aug on mixed@-6 over S1-S4, with the
  bootstrap s.d. as the standard error: pooled delta, 95% CI, tau^2, I^2. Done separately for DiffGate
  and SevGate.

## 5. Gate clamp sweep

Every DiffGate and SevGate model (S1-S4): the fusion weight is forced to r in {0, 0.25, 0.5, 0.75, 1}
(r = 1: F_local only; r = 0: F_ctx only) and compared with the learned r. Conditions: clean, mixed@-6,
unseen@-6, nstdb_all@-6. Seed-averaged scores. Reading: if some constant r matches the learned r within
0.002 on every condition, the gate is not acting as an input-dependent reliability selector.

## 6. Gate as a reliability score (selective prediction)

Each eval record is assigned to clean or corrupted with probability 1/2 (RNG seed 20261004, by record
id). The corruption is mixed|-6|whole (synthetic) or nstdb_mixed|-6|whole (real), and the two mixtures are
analysed separately. Scores of how unreliable a record is: -r (gated models, seed-averaged r), mean binary
predictive entropy over classes (every model), and the signal heuristic H1 (power 20-40 Hz / 0.5-20 Hz,
mean over leads, of the corrupted or clean input), along with random abstention (mean of 20 draws).
- Detection AUROC of each score for "corrupted".
- Macro-AUROC on the records kept at coverage 100, 90, 80, 70, 60, 50% (abstain on the most unreliable).
- Key comparison: SevGate abstaining by r vs Aug abstaining by entropy (gating is only worth its cost if
  r is a better abstention signal than an augmented model's own uncertainty). Descriptive.

## 7. Ablation summary table (descriptive)

Aug-only (Aug), gating-only (DiffGate-clean), aug+gating (DiffGate, SevGate) relative to Clean in every
setting, per noise family at -6 dB, per seed (mean +- s.d.).
