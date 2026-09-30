# PTB-XL Track 1 — Seed-0 Report (M1 vs M2)

Run date: 2026-09-29 · Code commit: `5426f6e` · Hardware: Apple M4 Pro (MPS) · Pipeline: `scripts/run_all.sh` (log: `logs/pipeline.log`)

## Summary

- Both models land at **~0.90 test macro-AUROC**, about **0.03 below** the published baselines (0.921–0.930).
- M2 (CNN + transformer) and M1 (CNN) are effectively tied: +0.003 test, +0.0001 val for M2.
- Neither model reached the agreed gate (best seed-0 test macro-AUROC ≥ 0.92), so **seeds 1–4 and the seed ensemble were not run**.
- Platt scaling (fitted on fold 9) cuts mean ECE from 0.129 to 0.023 without changing AUROC.
- Both models overfit. Validation AUROC peaks at epoch 10, and training stops early at epoch 20.

## 1. Data

PTB-XL v1.0.3, 100 Hz (`records100`), all 12 leads, input shape (12, 1000).
Preprocessing: 0.5–40 Hz 4th-order Butterworth band-pass (zero-phase), then a per-record, per-lead z-score.
Labels: every `scp_codes` key with `diagnostic == 1` is mapped to its `diagnostic_class` using `scp_statements.csv`. The task is multilabel over 5 superclasses.

| Split | Folds | Records (after drop) | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|---|---|
| Train | 1–8 | 17,084 | 0.445 | 0.256 | 0.245 | 0.229 | 0.124 |
| Val | 9 | 2,146 | 0.445 | 0.252 | 0.246 | 0.231 | 0.125 |
| Test | 10 | 2,158 | 0.446 | 0.255 | 0.241 | 0.230 | 0.121 |

- 411 of 21,799 records (36–44 per fold) have no superclass label and are **dropped**, as in Strodthoff et al. (2021). The remaining split sizes match the standard benchmark.
- No `patient_id` appears in more than one fold (18,869 patients checked).

## 2. Models and training

| | M1 | M2 |
|---|---|---|
| Backbone | Stem conv (k15, s2) + 4 stages × 2 residual SE blocks (k7), channels 32-64-128-128, stride 2 per stage → (128, 32) | Same as M1 |
| Sequence model | — | Sinusoidal PE + 2-layer transformer encoder (d_model 128, 4 heads, FF 256, dropout 0.1) |
| Pooling / head | Global average pooling → Linear(5) | Global average pooling → Linear(5) |
| Parameters | 1,046,557 | 1,311,517 |

Shared settings:
- Loss: BCE-with-logits with `pos_weight` = neg/pos from train label frequencies.
- Augmentation: mixup (α = 0.4).
- Optimizer: Adam (lr 5e-4, weight decay 1e-4), per-step cosine schedule with 5% warmup over 50 epochs.
- Batch size 32; early stopping with patience 10 on val macro-AUROC; seed 0.
- The test fold was evaluated once, using the checkpoint with the best val AUROC.

## 3. Results (fold 10, seed 0)

| Model | Test macro-AUROC | Val macro-AUROC | Best epoch / epochs run | Train time |
|---|---|---|---|---|
| M1 | 0.8967 | 0.8994 | 10 / 20 | 4.0 min |
| M2 | **0.8999** | 0.8995 | 10 / 20 | 5.2 min |
| xresnet1d101 (published) | 0.928 | | | |
| resnet1d_wang (published) | 0.930 | | | |
| inception1d (published) | 0.921 | | | |
| Ensemble (published) | 0.934 | | | |
| Naive | 0.500 | | | |

Per-class test AUROC:

| Model | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| M1 | 0.935 | 0.912 | 0.913 | 0.909 | 0.814 |
| M2 | 0.936 | 0.907 | 0.920 | 0.907 | 0.831 |
| Δ (M2 − M1) | +0.000 | −0.005 | +0.007 | −0.002 | +0.017 |

HYP is the weakest class for both models, and it is also the least prevalent (12%).

## 4. Calibration and operating point (M2, seed 0)

M2 was selected because it has the higher **val** macro-AUROC. All fitting and tuning used fold 9 only, and the results were applied unchanged to fold 10.

**Macro-AUROC:**
- 0.8999 both before and after Platt scaling, as expected, since Platt scaling is monotone.
- Patient-level bootstrap 95% CI (1,000 resamples): **[0.890, 0.908]**.

**ECE** (15 bins, per class):

| | NORM | MI | STTC | CD | HYP | Mean |
|---|---|---|---|---|---|---|
| Before | 0.076 | 0.112 | 0.117 | 0.097 | 0.242 | 0.129 |
| After Platt | 0.028 | 0.017 | 0.027 | 0.028 | 0.016 | **0.023** |

Before calibration, the probabilities are over-confident for the positive class. This is expected, because `pos_weight` shifts predictions upward.

**Per-class thresholds** (chosen to maximize F1 on fold 9) applied to fold 10:

| Class | Threshold | F1 | Precision | Recall | Specificity | Balanced accuracy |
|---|---|---|---|---|---|---|
| NORM | 0.59 | 0.848 | 0.798 | 0.906 | 0.815 | 0.860 |
| MI | 0.72 | 0.712 | 0.714 | 0.711 | 0.902 | 0.807 |
| STTC | 0.56 | 0.726 | 0.653 | 0.818 | 0.862 | 0.840 |
| CD | 0.63 | 0.746 | 0.757 | 0.736 | 0.930 | 0.833 |
| HYP | 0.76 | 0.458 | 0.470 | 0.447 | 0.930 | 0.688 |
| **Macro** | | **0.698** | **0.678** | **0.723** | | **0.806** |

## 5. Seeds 1–4 and seed ensemble

**Not run.** The agreed gate was: continue only if the best seed-0 test macro-AUROC is ≥ 0.92. The best result was 0.8999.

## 6. Diagnosis of the gap to the published results

- **Overfitting.** For M1, train loss falls steadily (0.86 → 0.49), while val AUROC peaks at epoch 10 (0.899) and then drifts down to about 0.88–0.89. Learning rate was still near its peak at that point (4.7e-4), so the cosine decay never took effect before early stopping.
- **Likely cause (not yet tested).** The published PTB-XL baselines train on **random 2.5 s crops** and average predictions over **sliding windows** at test time. This strong augmentation is absent from the current spec, which trains on the full 10 s record only. It is the most plausible source of the ~0.03 gap.
- **Other contributing differences.** The published baselines normalize with dataset-level standardization rather than a per-record z-score, and they apply no band-pass filter.

## 7. Recommended next step

Add random 2.5 s crop training and sliding-window test averaging to M1 and M2, then rerun seed 0. Each run takes about 10–20 minutes. If the gate is then passed, run seeds 1–4 and the seed ensemble as specified. The alternative is to accept the current numbers as the spec-faithful baseline and run seeds 1–4 anyway.

## Files

- `results/M1_s0.json`, `results/M2_s0.json`: metrics, parameter count, git hash, best epoch, and per-epoch history.
- `results/probs/M{1,2}_s0.npz`: val and test probabilities, labels, ecg_ids and patient_ids.
- `results/calibration_s0.json`: Platt parameters, ECE, thresholds and threshold metrics, bootstrap CI.
- `results/comparison_s0.txt`: comparison table.
- `logs/pipeline.log`: full run log, including the data-check output.
