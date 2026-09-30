# Reliability-Aware ECG Representation Learning (RACER) — Master Document

Target: Q1 journal submission. All code written from scratch.
This file merges `plan.md` (research plan) and `instructions.md` (Phase 1
experiment spec). Part B is the binding spec for the code being written now.

---

# PART A — RESEARCH PLAN

## A1. Executive Summary
Can a lightweight ECG model learn to distinguish diagnostically useful
morphology from unreliable signal content under degradation? The design uses
a large dataset (PTB-XL) for development and controlled experiments, then
independent smaller datasets for cross-dataset and real-world robustness.

The contribution is NOT generic multi-scale fusion or ECG+accelerometer
fusion. The central idea is **reliability-aware representation learning**:
estimate when local ECG morphology is trustworthy and adapt the
representation accordingly.

## A2. Core Research Question
Can an ECG model learn an explicit reliability representation that
identifies when fine-grained morphology is unreliable, improves prediction
under degradation, and transfers across datasets, lead configurations and
real-world recording conditions?

- RQ1: Does it maintain performance on clean clinical ECGs?
- RQ2: Does it degrade more gracefully than conventional encoders under
  controlled artifacts?
- RQ3: Does the mechanism transfer to an independent single-lead dataset?
- RQ4: Does predicted reliability track real-world quality in free-living ECG?

## A3. Dataset Strategy

| Dataset | Scale | Role | Purpose |
|---|---|---|---|
| PTB-XL 1.0.3 | ~21.8k 12-lead 10 s ECGs, 500 Hz | Main development | Training, controlled corruption, ablations |
| MIT-BIH NSTDB | bw / ma / em noise records, 360 Hz | Noise source | Real noise for augmentation and test |
| CinC2017 | 8,528 public single-lead ECGs, 300 Hz | External test | Single-lead / noisy robustness; AF transfer; "Noisy" class tests r |
| BUT QDB | 18 recordings / 15 subjects, 1000 Hz | Real-world test | Expert quality + accelerometer validation |
| Optional 4th | — | Stress test | Only if it adds a distinct acquisition shift |

Why a large main dataset: BUT QDB has only 15 subjects; treating millions
of windows as independent would be a statistical/leakage error. It is used
only for external validation.

PTB-XL notes:
- Official `strat_fold` splits are patient-level; use them.
- PTB-XL is not an expert quality dataset, but its metadata columns
  `baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems`
  flag real noise. Usable later as an in-dataset real-noise check.

## A4. Proposed Architecture (working name RACER)
Working name only, not a verified novelty claim.

```
ECG → Local morphology encoder (shallow 1D CNN) ─┐
                                                 ├→ Reliability module → Calibrated representation → Prediction
ECG → Long-context encoder (dilated TCN) ────────┘
```

Reliability module (per-timestep):
```
r     = sigmoid(g(F_local, F_ctx, |F_local − P(F_ctx)|))
F_cal = r · F_local + (1 − r) · F_ctx
```
The scientific distinction is modeling cross-scale disagreement/consistency
as reliability, not generic attention weights. Key risk: pathology
(ectopy, localized ST change) also causes local-vs-context disagreement, so
r must be shown to stay high on clean abnormal ECGs.

## A5. Training Objectives
- Task: multilabel BCE over PTB-XL superclasses.
- Auxiliary: severity-ordering ranking loss on r (λ2) and mild-pair logit
  consistency (λ1). Consistency is never applied to severe corruption, so
  clinically meaningful variation is not treated as noise.

## A6. Optional Motion-Conditioned Extension
BUT QDB accelerometer is used primarily as external validation (does r drop
during high motion and agree with expert quality?). Conditioning r on motion
is an extension, not the primary novelty, unless the literature audit shows
a clear gap.

## A7. Experimental Matrix (full paper)
- E0 — Go/no-go: compact encoders show a measurable failure mode under
  corruption, and RACER fixes it (= Part B).
- E1 — Clean PTB-XL benchmark.
- E2 — Single corruption × 3 severities.
- E3 — Mixed corruption.
- E4 — Architecture ablations.
- E5 — Cross-dataset: PTB-XL → CinC2017.
- E6 — Real-world quality/motion on BUT QDB.
- E7 — Reliability calibration and interpretability.

Ablations (full paper): remove reliability module; single-scale; remove
consistency feature; fixed reliability weight; remove auxiliary loss; no
corruption augmentation; augmentation-only baseline.

Additional baselines to add after Phase 1 (reviewer-expected): SQI-gate +
classifier (two-stage); denoising front-end + classifier; published PTB-XL
reference models (xresnet1d101, inception1d).

## A8. Cross-Dataset Evaluation
- **PTB-XL → CinC2017:** do not force incompatible label spaces. AF vs
  Normal is label-compatible (PTB-XL has AFIB) and can be tested directly;
  otherwise use a frozen-representation probe. Use CinC2017's "Noisy" class
  to test whether r detects bad signal. AliveCor approximates lead I, so a
  lead-I PTB-XL model is required (Phase 1 uses lead II — see D1).
- **PTB-XL → BUT QDB:** evaluate whether r tracks expert quality classes and
  accelerometer motion. Statistics per record/subject, aggregated across the
  15 subjects (or mixed-effects model). Faros lead is non-standard; expect
  morphology shift.

## A9. Metrics
- Macro-AUROC and macro-F1 (thresholds tuned on validation).
- Robustness drop = clean − corrupted.
- Performance-vs-severity curves.
- ECE if probabilistic predictions are reported.
- Ordinal discrimination / correlation between r and BUT QDB quality.
- Paired bootstrap CIs over patients/subjects.

## A10. Reliability Analysis
- r vs synthetic severity; r distributions clean vs corrupted.
- Does local reliability fall before broader temporal reliability?
- r on clean-abnormal (incl. ectopy: PVC, PAC) vs clean-NORM.
- BUT QDB: r vs expert quality, conditioned on motion.
- Representative examples where unreliable local evidence is suppressed.

## A11. Compute Plan
1–3M parameters; 1D CNN/TCN blocks; no transformers unless an ablation
justifies them; no foundation models, vision encoders or LLMs; mixed
precision; small, identical hyperparameter budget.
Hardware: Apple M4 Pro, PyTorch MPS (no CUDA).

## A12. Leakage and Statistical Design
Patient/subject grouping is mandatory. Corruption generated on the fly after
splitting. Noise excerpts disjoint between train and test. BUT QDB cycles
are never treated as independent subjects.

## A13. Strong Result vs Falsification
Strong: competitive clean performance; advantage grows with severity; beats
generic gate, multi-scale and consistency-loss-only controls; r tracks
corruption; transfers to CinC2017; r agrees with BUT QDB quality/motion.

Falsified if: CNN/TCN + augmentation matches the full model; generic gate
matches; r does not track degradation or real quality; gain appears for one
artifact type only; cross-dataset performance collapses; the literature
audit finds an essentially identical architecture.

## A14. Novelty Position
Multi-scale ECG SQA, adaptive fusion, ECG+ACC fusion and motion-aware ECG
analysis are active areas and are not claimed individually. The candidate
gap: using internal cross-scale disagreement to explicitly model morphology
reliability and calibrate the representation. Related concept to cite:
reliability-weighted stream fusion in audio-visual speech recognition.
Hypothesis only — must be re-audited before freezing the contribution.

## A15. Paper Structure
Introduction; Related Work (robust ECG, SQA, multi-scale ECG,
motion-aware ECG); Method; Experimental Setup; Main Results; Ablations;
Cross-Dataset Robustness; Reliability Analysis; Discussion & Limitations;
Conclusion.

Working titles:
- Learning When to Trust ECG: Reliability-Aware Representation Learning under Signal Degradation
- Cross-Scale Reliability Modeling for Robust ECG Representation Learning
- Learning to Suppress Unreliable ECG Morphology under Signal Degradation

## A16. Roadmap
1. Phase 1 experiment (Part B) → GO / NO-GO.
2. If GO: add reviewer baselines (A7), full ablations, 12-lead variant.
3. Lead-I model; CinC2017 transfer.
4. BUT QDB quality/motion analysis.
5. Statistics and reliability analysis.
6. Final novelty audit before freezing the contribution.

---

# PART B — PHASE 1 EXPERIMENT SPEC (binding for current code)

## B-scope. Reduced Phase 1 (decided 2026-09-29; overrides B2–B8 where they conflict)
User instruction: "PTB-XL only (100 Hz, lead II, official strat_fold split,
5 superclass multilabel). No NSTDB in phase 1; synthetic corruptions only.
Models: A (CNN), C (CNN+TCN concat), E (RACER-lite), each with and without
noise augmentation."

What this changes in the spec:
- B2.1: PTB-XL official `records100` files (100 Hz, 1000 samples). Folds are
  unchanged (1–8 / 9 / 10).
- B2.3: preprocessing is unchanged (0.5–40 Hz band-pass after corruption,
  z-score); 40 Hz is below the 50 Hz Nyquist.
- B2.4 and NSTDB in B3: not used. The NSTDB code stays in the repo for later
  phases.
- B3 families:
  - Train: synthetic baseline wander and synthetic EMG. EMG is band-limited to
    20–45 Hz, because 150 Hz does not exist at 100 Hz.
  - Unseen at test: electrode-motion bursts and dropout.
  - Powerline is dropped: 50 Hz is exactly the Nyquist frequency at 100 Hz.
  - Mixed test pool: baseline wander, EMG and motion burst.
  - Severities, modes and burst lengths are unchanged.
- B4: A, C and E are run. B, D, F and G stay implemented and are not run.
  At 100 Hz the stems use stride 2 (not 8) and the local branch has 3 blocks,
  so the receptive fields stay local ≈ 0.39 s and context ≈ 10.3 s. With the
  500 Hz stem unchanged, the local branch would span 1.6 s.
  Params: A 1.24M, C 1.09M, E 1.17M.
- B5: regimes clean and aug for each model × 5 seeds = 30 runs. The
  training protocol is otherwise unchanged.
- B6: no real-noise conditions. 28 test conditions; bootstrap compares E
  against augmented A and C.
- B8: the "real-noise" CI criterion cannot be evaluated. **Provisional**
  binding CI group is `mixed` only (`eval.binding_groups`), plus the
  unchanged r-monotonicity and clean-abnormal criteria. Unseen-family and
  0 dB CIs stay supporting evidence. Needs sign-off before results are read.
- B1 goal, reworded for this phase: does E beat augmented A and C under clean
  and synthetic-noise conditions on PTB-XL at 100 Hz?

## B0. Role
Build a small, reproducible PyTorch experiment. Clean, modular code. Do not
add features beyond this spec.

## B1. Goal
Does a reliability-aware two-branch ECG model beat a plain CNN/TCN trained
with the SAME noise augmentation, under clean, synthetic-noise and
real-noise conditions on PTB-XL?

## B2. Data
1. PTB-XL 1.0.3, 500 Hz records (100 Hz cannot represent 50 Hz powerline).
   Official folds: 1–8 train, 9 validation, 10 test. No custom split.
2. Task: 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP), multilabel.
   Drop records with no superclass label.
3. Input: full 10 s record, lead II only. Preprocessing: band-pass
   0.5–40 Hz, per-record z-score.
   Order: corruption is added to the RAW signal, THEN band-pass + z-score.
   Consequence: powerline (and part of EMG) is largely filtered out →
   powerline is a near-null sanity-check condition.
4. MIT-BIH NSTDB noise records `bw`, `ma`, `em`. Split each IN TIME: first
   60% for training augmentation, last 40% for test noise. Resample noise to
   500 Hz.

## B3. Corruptions (on the fly, after the split)
- Synthetic: baseline wander, Gaussian/EMG-like broadband noise, 50 Hz
  powerline, electrode-motion-like bursts, dropout.
- Real: NSTDB bw / ma / em.
- Mixed: two or more of the above.
- Severity via target SNR: 15, 6, 0 dB (mild / moderate / severe). In burst
  mode SNR is computed over the burst segment only.
- Dropout severity = duration {0.5, 1, 2} s, mapped to mild / moderate /
  severe.
- Two injection modes: whole-record and short bursts (2–4 s).
- Train/test family split:
  - Training augmentation: {NSTDB bw, NSTDB ma, synthetic baseline wander,
    synthetic EMG} and mixes of these.
  - Unseen at test: {NSTDB em, synthetic electrode-motion bursts, powerline,
    dropout}. Synthetic motion bursts are excluded from training because
    they resemble em.

## B4. Models (1–3M parameters each; report exact counts)
- **A. CNN-only:** 1D ResNet-style, 4–6 blocks.
- **B. TCN-only:** dilated, non-causal TCN, similar size.
- **C. CNN+TCN concat:** local CNN + context TCN, features concatenated.
- **D. Generic gate:** as C, `g = sigmoid(MLP([F_local, F_ctx]))`,
  `F = g·F_local + (1−g)·F_ctx`.
- **E. RACER-lite:** as D, gate also receives `|F_local − P(F_ctx)|`, P a
  linear projection:
  `r = sigmoid(g(F_local, F_ctx, |F_local − P(F_ctx)|))`,
  `F_cal = r·F_local + (1−r)·F_ctx`.
- **F. RACER-lite + losses:** for each clean ECG also feed a corrupted copy.
  Margin ranking loss so `mean(r_corrupted) < mean(r_clean) − margin`,
  weight λ2 (start 0.1). Mild-pair logit consistency between clean and
  mildly corrupted (15 dB) copies, weight λ1 (start 0.1). No consistency on
  severe corruption.
- **G. Control:** A + the same paired batches and mild-pair consistency (λ1)
  as F, no gate. Separates the gate's effect from the loss's effect.

Gates in D/E/F are per-timestep on time-aligned feature maps (both branches
output the same temporal length and channels); global average pooling after
fusion.

## B5. Training Protocol (identical for A–G)
- Regimes: (i) clean only; (ii) noise augmentation (random training-allowed
  family, random severity, p = 0.5). F and G run in regime (ii) only.
- AdamW, lr 1e-3, cosine schedule, batch 64, mixed precision (MPS autocast
  where supported, else fp32 — state which), max 40 epochs, early stop on
  CLEAN validation macro-AUROC for all models.
- 5 seeds per model/regime. Same augmentation code for every model.
- Minimal, identical hyperparameter budget.

## B6. Evaluation
- Test fold 10 only. Macro-AUROC and macro-F1 (per-class thresholds tuned
  on clean validation).
- Conditions: clean; each synthetic family × {15, 6, 0 dB}; each NSTDB type
  × {15, 6, 0 dB}; mixed; unseen-family hold-out; both injection modes.
- Report clean, corrupted and drop (clean − corrupted), mean ± std over
  seeds, and paired bootstrap over test patients (95% CI) for
  A/B/C/D/G vs E/F.
- Reliability diagnostics for D, E, F: mean r vs SNR per artifact; r on
  clean-abnormal (non-NORM) vs clean-NORM vs corrupted. Save plots.

## B7. Output
- Repo: `data.py`, `corruptions.py`, `models.py`, `train.py`, `eval.py`,
  config YAML, README with exact commands, `environment.yml` (conda env
  `medico`).
- `results/`: CSVs (model × condition × seed), performance-vs-SNR and
  r-vs-SNR figures.
- `summary.md` with the decision rule applied to the numbers.
- Test-only: a synthetic-data generator in the PhysioNet file formats, for
  end-to-end smoke tests before real data is available.

## B8. Decision Rule (verdict in summary.md)
- **GO** if E or F beats the best augmented A/B/C/D/G with a paired-bootstrap
  95% CI of the macro-AUROC difference excluding 0, under mixed and
  real-noise conditions, especially at 0 dB and on unseen families, AND
  mean r decreases monotonically with severity (15 > 6 > 0 dB), AND
  r(clean-abnormal) ≥ r(clean-NORM) − 0.05.
- **NO-GO / PIVOT** if augmented A/B/C/D/G match E/F (CI includes 0), or r
  does not track severity, or r drops on clean-abnormal ECGs. Then state that
  the outcome supports a benchmark-and-analysis paper instead.

## B9. Constraints
- Fixed seeds; log git hash and config with every run.
- No leakage: patient-level folds, noise generated after splitting, disjoint
  noise excerpts for train vs test.
- If ambiguous: pick the simplest option, state it in the README, continue.

---

# PART C — DATASETS AND DOWNLOAD

Datasets are provided after coding is complete. Links and versions verified
live on PhysioNet on 2026-09-29; all four are open access, and every version
listed is the latest.

| Dataset | Version | Landing page | Files needed |
|---|---|---|---|
| PTB-XL | 1.0.3 | https://physionet.org/content/ptb-xl/1.0.3/ | `ptbxl_database.csv`, `scp_statements.csv`, `records500/` |
| NSTDB | 1.0.0 | https://physionet.org/content/nstdb/1.0.0/ | `bw`, `ma`, `em` (`.dat` + `.hea`) |
| CinC2017 | 1.0.0 | https://physionet.org/content/challenge-2017/1.0.0/ | `training2017.zip`, `REFERENCE-v3.csv` (latest labels) — Phase 2 |
| BUT QDB | 1.0.0 | https://physionet.org/content/butqdb/1.0.0/ | 18 record folders (`100001/` … `126001/`) — Phase 2 |

Download (AWS CLI is installed on this machine; `wget` is not):
```
# PTB-XL (~3 GB)
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ data/ptb-xl/

# NSTDB — only the three noise records (not on the S3 mirror; use HTTPS)
mkdir -p data/nstdb && for f in bw ma em; do for e in dat hea; do
  curl -fL -o data/nstdb/$f.$e https://physionet.org/files/nstdb/1.0.0/$f.$e; done; done

# CinC2017 (Phase 2)
aws s3 cp --no-sign-request s3://physionet-open/challenge-2017/1.0.0/training2017.zip data/cinc2017/
aws s3 cp --no-sign-request s3://physionet-open/challenge-2017/1.0.0/REFERENCE-v3.csv data/cinc2017/

# BUT QDB (Phase 2, large)
aws s3 sync --no-sign-request s3://physionet-open/butqdb/1.0.0/ data/butqdb/
```
The wget alternative from dataset.md (`wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/`)
also works if wget is installed, but it nests files under
`physionet.org/files/...`; the data paths are configurable in the YAML.

---

# PART D — OPEN ITEMS

- D1. Phase 1 uses lead II; CinC2017 transfer needs a lead-I model
  (AliveCor ≈ lead I). Train a lead-I variant in Phase 2.
- D2. Final novelty audit before freezing the contribution.
- D3. Use PTB-XL noise metadata columns as an in-dataset real-noise check
  (Phase 2).
- D4. NSTDB `em` record reportedly contains a persistent regular rhythm
  (residual ECG) in both channels (bioRxiv 10.1101/2022.10.18.512701).
  em is an unseen test family: state this limitation, cite the paper, and
  also report the unseen-family result with em excluded.
- D6. B-scope (2026-09-29): NSTDB real-noise evaluation, 500 Hz and variants
  B/D/F/G are deferred, not dropped. Re-enable via `configs/phase1.yaml`
  (`data.fs`, `corruptions.*`, `variants`, `eval.binding_groups`).
- D7. Noise calibration (open): corruption is added before the band-pass, so
  nominal SNR ≠ in-band SNR. At 100 Hz, nominal 0 dB gives these median
  in-band SNRs (400 real lead-II records):
  - baseline wander: 23 dB whole-record, 9 dB burst
  - EMG: 1 dB
  - motion burst: 1–3 dB
  - baseline wander + EMG mix: 3–4 dB
  Only baseline wander remains near a no-op. Decide whether to calibrate SNR
  in-band before the real runs.
- D5. Dataset versions checked 2026-09-29: PTB-XL 1.0.3, NSTDB 1.0.0,
  CinC2017 1.0.0 (labels REFERENCE-v3), BUT QDB 1.0.0 — all latest; no
  newer releases found on PhysioNet or GitHub.
