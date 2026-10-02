Reliability-Aware ECG Representation Learning: Research Plan v2 with SOTA References

Sep 29, 2026

Version 2 keeps the PTB-XL to CinC2017 to BUT QDB design, adds a tiered set of SOTA papers to compare against, and flags one near-overlapping paper that must be addressed before the novelty claim is frozen.

What changed in v2
The literature search changed the plan in six places; each is expanded in the section named.

1. SOTA reference set added. Baselines are now tiered: PTB-XL benchmark models, foundation models, published robustness protocols, and quality-assessment methods (see SOTA papers).
2. A near-overlapping paper was found. An arXiv paper already evaluates lightweight (at most 512k parameters) noise and out-of-distribution handling on PTB-XL and BUT QDB with real electrode-motion noise (see Novelty and overlap risk).
3. A direct warning about gating. A 2026 ECG-PPG study found that adaptive gates detected total modality loss but barely tracked graded quality. Your reliability score r may do the same, so a supervised or calibrated signal is now recommended.
4. Real noise replaces hand-made noise. MIT-BIH Noise Stress Test recordings (baseline wander, muscle, electrode motion) are added to the synthetic corruption benchmark.
5. CinC2017 has a Noisy label. It is a four-class task (normal, AF, other, too noisy), so it gives a second real-world quality signal, not only a rhythm test.
6. BUT QDB now has an official protocol. A 2026 Scientific Data descriptor adds a standardized evaluation protocol and open-source benchmarking code; use it instead of a custom split.
SOTA papers to compare against
Compare against papers in five groups. Do not try to beat large foundation models on clean accuracy; use them as reference points and compare robustness, parameter count and reliability behavior instead.

A. PTB-XL clean-performance benchmarks
Paper

Use it for

Note

Strodthoff et al., Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL
Task definitions, splits, AUROC protocol, CNN baselines

Resnet- and inception-style CNNs were strongest; xresnet1d101 was best across tasks. Code is public (helme/ecg_ptbxl_benchmarking).

Reproduction of the PTB-XL benchmark, medRxiv 2025
Prior noise-robustness numbers on the same seven models

Adds Gaussian noise to test ECGs and tracks the AUROC drop; reuse this as your simplest robustness comparison.

Accurate overall, uneven by patient, medRxiv 2026
Current clean ceiling on the five superclasses

Reports macro AUC near 0.92 (best reported around 0.925); a preprint, so cite with care.

B. ECG foundation models (reference tier, not size-matched)
Paper

Size

Why include it

HuBERT-ECG
30.5M to 188.6M parameters

Pretrained on 9.1M ECGs; includes single-lead benchmarks, which helps the CinC2017 comparison.

ECG-FM
Transformer, 1.5M pretraining ECGs

Open weights and code.

ST-MEM
About 85M

Works with limb and single leads; evaluated on PTB-XL, so a natural reference.

ECG-JEPA
About 85M

Already reports PTB-XL results under baseline-drift and powerline noise at three levels; mirror its protocol so your curves are comparable.

BenchECG / xECG
Benchmark

Re-evaluates ST-MEM, ECG-JEPA and ECGFounder with official weights in one framework.

FOUND-AF
Benchmark

Compares nine foundation models on AF, including a small multi-scale convolutional model (CLEF Small, 0.4M parameters) that is the closest size-matched reference.

C. Noise-robust ECG classification (closest to your claim)
Paper

Use it for

Note

Enhancing ECG Classification Robustness with Lightweight Unsupervised Anomaly Detection Filters
Direct comparator and overlap check

Uses PTB-XL, BUT QDB and NSTDB electrode-motion noise with models at most 512k parameters.

Negative-ResNet
Published noisy-ambulatory-ECG method

Read the full text to confirm it can be re-run on PTB-XL.

Uncertainty-Aware Multi-view Arrhythmia Classification
NSTDB protocol

Adds NSTDB noise at 15 to 0 dB SNR.

Tiny Transformer for Low-Power Arrhythmia Classification
SNR scaling formula

Gives the noise-scaling equation for NSTDB electrode motion.

D. Adaptive or quality-aware fusion (closest to the architecture)
Paper

Use it for

Note

CardioFusion-AI
Fairness protocol and a cautionary result

Compares eight fusion strategies over five seeds; gates detected full modality loss but barely tracked graded quality.

From Motion Artifacts to Clinical Insight
ECG plus accelerometer gated fusion

Already publishes gate analysis, so do not claim ECG+ACC gating as novel.

High-Reliability Signal Quality Validation
Hand-crafted motion and impedance gating

Non-learned comparator for motion-informed quality.

E. Signal quality assessment and CinC2017
Paper

Use it for

Note

BUT QDB data descriptor, Scientific Data 2026
Official evaluation protocol and code

18 recordings from 15 healthy subjects, over 86 hours of expert labels, three quality classes, plus accelerometer.

Deep learning SQA in wearable ECG (Ma et al., CinC 2023)
Supervised SQA baseline on BUT QDB

xResNet models; the first reports 98.87% sensitivity and 99.83% specificity for usable versus unusable.

Self-supervised learning for ECG SQA
Representation-based SQA baseline

SimCLR, BYOL and SwAV on BUT QDB; SwAV was best.

CinC2017 challenge paper
Reference scores

Four teams tied at F1 0.83 and the top 11 were within 2%; a 45-algorithm ensemble reached 0.87.

BIT-CNN on CinC2017
Modern single-model reference

F1 of 81.75% (normal, AF, other); notes that strong results sit in the early-to-mid 80s.

Andreotti CinC2017 code
Open CNN and feature baselines

Ready-to-run reference implementation.

Two caveats apply. The CinC2017 hidden test set is not public, so any score you report on the 8,528 public recordings is not directly comparable to challenge scores. And one SQA paper notes that BUT QDB Class B is dominated by baseline-drift noise, so quality results there will not represent motion or muscle noise well (Swin-GAN SQA paper).

Novelty and overlap risk
The defensible claim is narrow: a cross-scale consistency signal that sets how much local morphology is trusted, validated against real quality labels. Three found papers sit close to it and each must be cited and differentiated.

• Closest overlap: lightweight unsupervised anomaly-detection filters. Same datasets (PTB-XL, BUT QDB, NSTDB electrode-motion noise) and the same small-model budget. It acts as an upstream filter that rejects noisy or out-of-distribution inputs. Your difference must be that reliability changes the representation inside the model rather than rejecting the input. State that explicitly and include the filter as a baseline.
• Gate-tracking evidence: CardioFusion-AI. Learned gates, including ones conditioned on a quality index, did not follow graded quality when both inputs were present. This is the strongest argument that r will not learn reliability from the task loss alone. Plan for it: add a quality-supervised or calibrated variant, and report gate-versus-severity honestly.
• ECG plus accelerometer gating: attention-gated fusion, Sensors 2026. Gate values already reported per noise and motion condition. Keep motion conditioning as a small extension, not the contribution.
• Multi-scale small ECG models: CLEF Small (0.4M parameters). A multi-scale convolutional foundation model at your size. Check whether its multi-scale objective already covers cross-scale agreement.
Open audit items: search for cross-scale or local-versus-global consistency used as a reliability weight in ECG or PPG, and read Negative-ResNet in full. The exact term "reliability-aware" is not a novelty claim on its own.

Research question and datasets
The core question is unchanged: can a compact ECG model learn an explicit reliability signal that flags unreliable local morphology, improves prediction under degradation, and transfers across datasets and recording conditions? It splits into four testable questions.

1. Does reliability-aware learning hold clean PTB-XL performance?
2. Does it degrade more gracefully than conventional encoders under controlled and real noise?
3. Does the mechanism transfer to an independent single-lead dataset?
4. Does predicted reliability follow real expert quality labels and motion?
Dataset

Scale

Role

Change in v2

PTB-XL

21,837 12-lead 10 s ECGs, about 18,885 patients

Main development

Use the official benchmark folds and label tasks so results are comparable; confirm patient-level separation in the dataset audit. Train a 12-lead model and a single-lead variant for transfer.

MIT-BIH NSTDB

Three 30-minute noise recordings (baseline wander, muscle, electrode motion)

Real noise source

New. Inject into PTB-XL after the split (see Corruption benchmark).

CinC2017

8,528 public recordings, 9 to 61 s

External single-lead test

Use AF versus non-AF and the Noisy class; the hidden test set is unavailable.

BUT QDB

18 recordings, 15 healthy subjects, 1,000 Hz ECG plus 100 Hz accelerometer

Real-world reliability validation

Follow the official protocol; treat subjects, not cycles, as the unit of analysis.

Optional fourth

Not chosen

Distribution-shift stress test

No candidate was verified in this search; add only with a clear evaluation question.

Label formulation. Use the five PTB-XL diagnostic superclasses as the primary multilabel task, since published clean results (macro AUC about 0.92) give a clear ceiling. Add one rhythm task that includes AF so the CinC2017 transfer test has a compatible label.

Corruption benchmark
Build the benchmark from both synthetic artifacts and real NSTDB noise, applied only after the split, at three or more severity levels. Real noise matters because papers that add only white noise to PTB-XL are criticized as unrepresentative (denoiser study).

Artifact

Synthetic version

Real-noise version

Baseline wander

Low-frequency drift

NSTDB baseline-wander recording

Muscle (EMG)

Broadband noise

NSTDB muscle-artifact recording

Powerline

50 and 60 Hz component

None

Electrode motion

Transient bursts and amplitude changes

NSTDB electrode-motion recording

Dropout or masking

Short segments removed or distorted

None

Mixed

Two or more of the above

Real noise plus one synthetic artifact

Design rules:

• Severity by SNR. Scale noise to a target SNR using the formula in the tiny-transformer paper. A range such as 15, 6 and 0 dB matches the 15 to 0 dB used in prior NSTDB work.
• Short segments, not whole records. Real motion and muscle artifacts appear briefly, so inject bursts of a few seconds; this also tests local reliability.
• Disjoint noise excerpts. NSTDB has only about 30 minutes per noise type. Use different time ranges of each recording for training augmentation and for testing, or the test noise is not truly unseen.
• Comparable curves. Reproduce the baseline-drift and powerline levels of the ECG-JEPA protocol and the Gaussian-noise curve of the PTB-XL reproduction so your numbers sit next to published ones.
• Hold one family out. Train with some artifact families and test on a held-out one; a gain on a single family is one of your own falsifiers.
Report clean score, corrupted score and relative drop for every artifact, severity and model.

Architecture and training
RACER (Reliability-Aware Cross-Scale Representation, a working name and not a verified novelty claim) keeps the two-branch design, but v2 adds an explicit way to make the reliability score mean something. The model stays at 1 to 5M parameters with CNN and TCN blocks; add transformers only if an ablation justifies them.

 1. Local morphology branch. A shallow 1D CNN for QRS and local P and T shape.
 2. Long-context branch. A compact dilated TCN for rhythm and beat-to-beat context.
 3. Reliability module. Estimates agreement between the two branches and sets how much local morphology to trust.
 4. Prediction head. Reads the calibrated representation.
r = \sigma\big(g(F_{local},\, F_{ctx},\, |F_{local} - P(F_{ctx})|)\big)

F_{cal} = r \cdot F_{local} + (1 - r) \cdot F_{ctx}
Making r mean reliability (new). With only a task loss, the gate can learn any weighting that lowers loss, and a recent study found gates did not follow graded quality (CardioFusion-AI). Test three options, each as its own ablation:

• Task loss only. The current plan; the control.
• Severity-ordering loss. Because you inject the noise, you know severity; add a ranking penalty so r for a heavily corrupted copy is lower than for the clean copy of the same ECG.
• Consistency on mild pairs. Encourage stable outputs between a clean ECG and a mildly corrupted copy, and do not apply it to severe corruption, so real abnormal morphology is not treated as noise.
Training objective: L = L_task + λ1 · L_consistency + λ2 · L_severity, with the task loss as cross-entropy or multilabel BCE. Use five seeds per model, as the fusion study did, and a small hyperparameter sweep with mixed precision.

Motion extension (optional). Condition r on accelerometer intensity only on BUT QDB, and treat it as an add-on. ECG plus accelerometer gating is already published, so it is not the contribution.

Baselines, ablations and experiments
The fairness control that decides the paper is an augmentation-only baseline trained on the same corruption families and the same real-noise excerpts as the full model.

Baselines
Baseline

Purpose

Reference

Compact 1D CNN

Size-matched ECG-only

Own

Compact TCN

Size-matched ECG-only

Own

Same models plus corruption augmentation

Can augmentation alone explain the gains?

Own; see augmentation study on CinC2017

Multi-scale without reliability

Separates reliability from multi-scale fusion

Own; compare CLEF Small

Generic attention or gate, plus a quality-index-conditioned gate

Separates consistency from generic gating

Own; design follows CardioFusion-AI

xresnet1d101

Published PTB-XL reference (larger)

Strodthoff et al.
Anomaly-detection filter plus classifier

Input rejection versus representation-level reliability

Rifet Ibrahim et al.
Quality classifier as a pre-filter

Published SQA used as a gate

Ma et al., CinC 2023
Foundation-model reference (one or two)

Upper reference; report parameter count

ST-MEM, ECG-JEPA or HuBERT-ECG

Ablations
Ablation

Question

Remove reliability module

Does reliability modeling matter?

Single-scale ECG

Does cross-context information matter?

Remove the disagreement input

Is explicit agreement useful beyond a generic gate?

Fixed reliability weight

Is dynamic adaptation needed?

Task loss only for r

Does r become reliability without extra supervision?

Remove severity-ordering loss

Does the auxiliary objective matter?

No corruption augmentation

Does robustness come from architecture?

Experiment matrix
15. E1 Clean PTB-XL benchmark on the official folds.
16. E2 Single synthetic corruption at three or more severities.
17. E3 Real NSTDB noise and mixed corruption, with held-out artifact families.
18. E4 Architecture and objective ablations.
19. E5 CinC2017: AF versus non-AF, plus the Noisy class as a quality signal.
20. E6 BUT QDB: reliability versus expert quality class and motion, using the official protocol.
21. E7 Reliability calibration, gate-versus-severity curves and examples.
22. E8 Clean-abnormal check: r should stay high on clean pathological ECGs.
Metrics, reliability analysis and success criteria
The most testable prediction comes from how BUT QDB defines quality: local morphology reliability should fall at Class 2, and context reliability only at Class 3. Class 1 has all waves clearly visible; Class 2 still allows reliable QRS detection but not reliable wave onsets and offsets; Class 3 does not allow reliable QRS detection (BUT QDB). That maps directly onto a local-versus-context reliability split.

Metrics
• PTB-XL: macro AUROC (the benchmark's standard) and macro-F1.
• CinC2017: F1 averaged over normal, AF and other, as the challenge defines it.
• Robustness: clean minus corrupted score, plus performance-versus-severity curves.
• Calibration: expected calibration error when probabilities are reported.
• Reliability versus quality: rank correlation and pairwise class discrimination between r and BUT QDB class.
• Uncertainty: five seeds; bootstrap over subjects on BUT QDB (n = 15) and over patients on PTB-XL.
Reliability analysis
23. Plot r against synthetic and NSTDB severity, per artifact.
24. Compare r for clean, corrupted and clean-abnormal ECGs.
25. Check whether local reliability falls before context reliability, on synthetic noise and on BUT QDB Class 2 versus Class 3.
26. Compare r with the CinC2017 Noisy label.
27. Stratify by accelerometer intensity on BUT QDB.
28. Show examples where unreliable local evidence is suppressed, and examples where it fails.
What counts as a strong result
• Clean performance within noise of the size-matched baselines.
• An advantage that grows with severity, including on held-out artifact families and real NSTDB noise.
• A win over generic attention, quality-conditioned gating and the augmentation-only baseline.
• r that orders BUT QDB classes and does not drop on clean abnormal ECGs.
What would falsify the idea
• CNN or TCN plus augmentation matches the full model.
• Generic or quality-conditioned gating performs equally well.
• r fails to track severity or expert quality.
• The gain appears for one artifact type only.
• Cross-dataset performance collapses, or the overlap audit finds an equivalent architecture.
Note that the intermediate class (called Class B in that paper) is reported to consist mainly of baseline-drift noise (Swin-GAN SQA paper), so state this limit when generalizing.

Roadmap
Run the augmentation-versus-architecture kill test before building anything else; move the overlap audit up front instead of leaving it last.

 1. Dataset audit. Confirm versions, licences, patient IDs and official folds for PTB-XL; get NSTDB, CinC2017 and the BUT QDB protocol code.
 2. Overlap audit. Read the anomaly-detection filter paper, Negative-ResNet and CLEF in full; search for cross-scale consistency used as a reliability weight.
 3. Reproduce baselines. Compact CNN and TCN, plus xresnet1d101 on the official folds to confirm your pipeline matches published numbers.
 4. Build the corruption benchmark. Synthetic families and NSTDB noise with disjoint train and test excerpts.
 5. Kill test. Compare CNN or TCN plus augmentation against the smallest reliability module under mixed and real-noise corruption. If the gap is negligible, pivot to a benchmark-and-analysis paper on how compact ECG models fail under degradation.
 6. Make r meaningful. Add the severity-ordering and mild-pair consistency losses; check r against severity and the clean-abnormal set.
 7. External tests. CinC2017 (AF and Noisy class), then BUT QDB with the official protocol.
 8. Statistics and reliability analysis. Five seeds, subject-level bootstrap, calibration.
 9. Freeze the contribution only after a final literature check, since this area is moving quickly.
Sources and verification
Every paper above was found through web search on 29 Sep 2026 and summarized from search excerpts; the full texts were not opened. Read each paper before citing it, and check these items first:

• Parameter counts for HuBERT-ECG, ST-MEM, ECG-JEPA, ECGFounder and CLEF, taken from the FOUND-AF comparison table.
• The macro AUC ceiling of about 0.925 on PTB-XL superclasses, from a 2026 preprint that is not peer reviewed.
• The CinC2017 scores (challenge paper, BIT-CNN) and the BUT QDB sensitivity and specificity (Ma et al.).
• Whether Negative-ResNet is a usable, re-runnable baseline; only its reference list was seen.
• Whether the anomaly-detection filter paper's arXiv version has been published in a journal, which affects how it is cited.
Dataset pages: BUT QDB (PhysioNet), CinC2017 challenge, PTB-XL benchmark repository.
