# Implementation decisions (Phase 1)

Decisions for points the spec (`docs/master.md`) left open.

Training (`train.py`, `configs/phase1.yaml`)
- AdamW, lr 1e-3, weight decay 0.01 (the AdamW default), batch 64, max 40 epochs.
  Cosine decay is applied per step over the full 40-epoch budget, with no warmup and a floor of 0.
  The budget is identical for all variants, and early stopping just truncates the schedule.
- Early stopping on clean validation macro-AUROC (fold 9) for every variant, patience 8 epochs.
  The checkpoint with the best score is kept. The comparison is strict (`>`), so ties keep the
  earlier epoch.
- **Precision: fp32.** On MPS, fp16 autocast gave no speedup: it was 0–6% slower than fp32 for
  every model (e.g. racer 116.5 vs 109.4 ms/iter at batch 64). `train.amp` is therefore `false`.
  The fp16 path still exists (`amp: true`): it uses a GradScaler, and a one-time
  forward/backward/step check on a throwaway model falls back to fp32 if the check fails.
  `meta.json` records the dtype used (`amp_dtype`, `amp_note`).
- Paired variants (F, G): a batch holds 64 ECGs, each seen as a clean view and a corrupted view
  (128 views). Both views go through one forward pass, so BatchNorm sees the same 50/50
  clean/corrupted mix as regime aug. BCE is averaged over both views. The rank loss (F only) and
  the mild-only consistency loss are exactly as in `docs/CONTRACT.md`. The consistency term is exactly
  0 when a batch has no mild pair.
- Regime `aug` corrupts each training ECG with p = 0.5 via `Corruptor.sample_train`. The family
  is one of `corruptions.train_families` (Phase 1: baseline wander, EMG), or with p = 0.25 a
  2-family mix; severity and mode are uniform. NSTDB (train split, first 60% in time) is loaded
  only if an `nstdb_*` family is configured. The clean regime uses no corruptor.
- Clean validation signals are preprocessed once. `thresholds.json` holds per-class F1-optimal
  thresholds (predict positive iff p ≥ t). They are tuned on clean validation using the best
  checkpoint. Candidate thresholds are the distinct predicted scores, and ties go to the lowest
  threshold. A class with no positives gets 0.5.
- Seeding: `random`, `numpy` and `torch` (CPU+MPS), plus a DataLoader generator and
  `data.worker_init_fn`. Workers are persistent (`num_workers: 4`), because macOS spawn start-up
  and pickling of X_raw are expensive. MPS kernels are not guaranteed to be bit-deterministic,
  and `torch.use_deterministic_algorithms` is not enabled.
- Provenance: `meta.json` holds the `git rev-parse HEAD` hash (or `"nogit"` when there is no
  commit) and a dirty flag. It also records the parameter count, device, AMP dtype, epochs,
  best epoch, best validation AUROC and wall time. The resolved config is saved as `config.yaml`.
  `meta.json` is written last, so an interrupted run is rerun from scratch.

Evaluation (`eval.py`)
- Test set: fold 10 only. Phase 1 test corruptions are synthetic, generated with fixed per-record
  seeds (below). When NSTDB families are configured, test noise comes from the NSTDB **test**
  split (last 40% in time), so train and test noise are disjoint.
- Fixed test noise: each record's generator is seeded from (`eval.noise_seed`, family, mode,
  record index). Every model and seed therefore sees identical corrupted test signals. The seed
  does not depend on severity, so the three severities of a family/mode share the same excerpt,
  channel and burst position and differ only in scale. These are common random numbers, which
  pair the severity curves. Each condition's corrupted and preprocessed test set is computed once
  and cached under `data/cache/eval_test/`. The cache key covers the seed, the test ecg_ids and
  the source of `data.py` and `corruptions.py`.
- Conditions: clean, then every configured family (train + unseen) × {mild 15 dB, moderate 6 dB,
  severe 0 dB} × {whole, burst}. Dropout does not depend on mode, so it is evaluated once per
  severity (0.5/1/2 s). Then `mixed` × 3 severities × 2 modes. Phase 1 has 28 conditions.
- **Mixed test condition:** for each record, draw 2 distinct families from all configured
  *additive* families (Phase 1: baseline wander, EMG, motion burst). The two noises are summed at unit power, and the sum is scaled
  to the target SNR (CONTRACT). Dropout is excluded because a mix is a sum of noises and
  `Corruptor.apply` rejects dropout inside a family list. The mixed pool therefore includes
  families that were unseen in training.
- Aggregate groups: `seen_families` (train families), `unseen_families`, `mixed` and
  `all_corrupted`. `real_noise` (NSTDB) and `unseen_no_em` (docs/master.md D4) exist only when NSTDB
  families are configured, so Phase 1 does not have them. Each group is reported per severity and pooled ("all"). A group's
  metric is the unweighted mean of its member conditions' metrics. Drop = clean − corrupted,
  computed per run and then summarised as mean ± std over seeds (ddof=1).
- Inference runs in fp32. Macro-AUROC is the unweighted mean of per-class AUROC; a class that
  lacks positives or negatives in a sample is skipped. Macro-F1 uses each run's
  `thresholds.json`.
- Paired bootstrap:
  - Resamples test **patients** (clusters with all their records) with replacement: 1000
    resamples, 95% percentile CI.
  - Uses the same resamples for every model and condition, so the comparison is paired.
  - The statistic is the macro-AUROC difference of **seed-averaged predicted probabilities**
    (the mean over the 5 seeds' probabilities, per model).
  - Compares each candidate (E, F) that has runs against each augmented baseline (A, B, C, D,
    G) that has runs, for every group and severity. In Phase 1 that is E vs A and E vs C.
- Reliability diagnostics (D, E, F): the mean r per record is `r` averaged over time. The mean
  over records is reported per run and condition (`reliability.csv`).
  Clean-NORM means NORM is the *only* label. Clean-abnormal means no NORM label.
- **B8 decision rule, operationalised before any result is seen:**
  - The best augmented baseline for a group is the one among the augmented baselines with runs
    (Phase 1: A, C) with the highest seed-averaged macro-AUROC on that group.
  - Binding criteria, for a candidate M (Phase 1: E):
    1. For every group in `eval.binding_groups`, the 95% CI of AUROC(M) − AUROC(best baseline)
       has a lower bound > 0, pooled over severities and modes. **Phase 1: `[mixed]` only.**
       The original rule also required `real_noise`, which needs NSTDB. This is provisional and
       still needs sign-off.
    2. Mean r over all corrupted conditions at each severity (all families including mixed, both
       modes, averaged over seeds) strictly decreases: mild > moderate > severe.
    3. r(clean-abnormal) ≥ r(clean-NORM) − 0.05.
  - The verdict is GO iff some candidate passes all binding criteria. Otherwise it is NO-GO /
    PIVOT (benchmark-and-analysis paper), or INCOMPLETE if runs are missing.
  - The following are printed as *supporting, non-binding* criteria: the 0 dB (severe) CI for
    mixed, and the unseen-family CIs (all and severe). With NSTDB configured, real-noise and
    unseen-without-em CIs are added.
- `summary.md` is written to `results/summary.md` (`results/smoke/summary.md` for `--smoke`).
- NSTDB `em` limitation (docs/master.md D4; only relevant once NSTDB is configured): em reportedly
  contains residual ECG (bioRxiv 10.1101/2022.10.18.512701). The unseen-family result is then
  also reported without em.
- Powerline (500 Hz only) is mostly removed by the 0.5–40 Hz band-pass applied after corruption,
  so it is a near-null sanity check. At 100 Hz it is refused, because 50 Hz is the Nyquist
  frequency.

Data, corruption and model decisions (`data.py`, `corruptions.py`, `models.py`)
- Label mapping, preprocessing, the NSTDB split and resampling, and synthetic families: see
  `docs/CONTRACT.md`. The other modules' authors append their further decisions here.

### Data and corruption decisions (data.py, corruptions.py)
- **Sampling rate (Phase 1): 100 Hz from the official PTB-XL `records100` files (`filename_lr`),
  1000 samples per record.** These are the files the published PTB-XL 100 Hz benchmarks use; we
  do not decimate `records500` ourselves. `data.fs: 500` switches back to `records500`. The
  signal cache is keyed by fs.
- Labels: every `scp_codes` key with `diagnostic == 1`, any likelihood (incl. 0), mapped via `diagnostic_class` (official PTB-XL example). Records with no superclass are dropped.
- Band-pass: 4th-order Butterworth 0.5–40 Hz, zero-phase (`sosfiltfilt`); z-score `(x − mean)/(std + 1e-6)`, computed in float64. Measured attenuation: 50 Hz −17.9 dB, 60 Hz −30.8 dB.
- NSTDB: split in time at 360 Hz (first 60 % train / last 40 % test), then each part resampled separately (`resample_poly` 25/18), so no resampling-filter leakage across the split.
- SNR: each family's noise is made zero-mean, unit-power over the affected segment; mixes are summed, re-centred, then scaled. Signal power = variance of the clean segment (on the RAW signal, before filtering).
- Bursts: length uniform 2–4 s (integer samples), uniform position; signal outside the burst is untouched.
- Synthetic families: baseline wander = 1–3 sinusoids 0.05–0.5 Hz; EMG = white noise band-passed 20 Hz to min(150 Hz, 0.45·fs), i.e. 20–45 Hz at 100 Hz; powerline = 50 Hz + 100 Hz harmonic (rel. amp 0.05–0.2); motion burst = 1–10 Hz band-limited noise + step under a Tukey(0.5) envelope (2–4 events in whole mode).
- Dropout: `round(sec × fs)` raw samples set to 0 mV; mode-independent; cannot be mixed.
- `sample_train`: mix of 2 train families with p = 0.25; refuses a test-split Corruptor (leakage guard).
- **Known issue (pending decision):** because corruption is applied before the 0.5–40 Hz band-pass (B2), nominal SNR ≠ in-band SNR.
  - At 500 Hz with NSTDB, nominal 0 dB gave an effective post-filter SNR of about 27 dB (synthetic baseline wander), 19 dB (NSTDB bw), 12 dB (powerline), 8 dB (EMG), 5 dB (NSTDB ma) and 3 dB (NSTDB em, motion burst).
  - **At 100 Hz (Phase 1)**, measured on 400 real records100 lead-II records, the median in-band SNR at nominal 0 dB is:
    - baseline wander: 23.3 dB whole-record, 8.9 dB burst
    - EMG: 0.9 dB whole, 1.0 dB burst
    - motion burst: 3.1 dB whole, 1.1 dB burst
    - baseline wander + EMG mix: 3.8 dB whole, 3.3 dB burst
  - EMG is now almost entirely in band. Baseline wander is still largely removed by the filter.

### Model decisions (models.py)
- **100 Hz (Phase 1):** every stem uses strides (1, 1, 2), a total stride of 2, so T' = 500 (50 Hz). The local branch has 3 blocks (RF 39 samples = 0.39 s) and the context branch 7 blocks (RF 1031 samples = 10.3 s). A (cnn) has RF 2.91 s and T' = 125. Params: A 1,235,237; B 1,255,709; C 1,093,189; D 1,141,957; E 1,174,853. Reason: the 500 Hz stem (stride 8), applied unchanged at 100 Hz, would give the "local" branch a 1.6 s RF (about two beats) and the context branch a 41 s RF, which breaks the local/context split. `models.ARCH` holds both settings; everything below the stem is unchanged.
- The rest of this section describes the **500 Hz** setting.
- Every branch: its own strided stem (3 × Conv-BN-ReLU, kernels 7/5/5, stride 2 each), total stride 8 → T' = 625 (62.5 Hz). Residual blocks = 2 convs + BN, identity/1×1 shortcut, non-causal "same" padding. BatchNorm everywhere, no dropout.
- Local branch: 4 blocks, kernel 3, no dilation, RF 159 samples (0.32 s). Context branch: 7 dilated blocks (dilation 1…64), RF 4095 samples (8.19 s). C/D/E use exactly these two branches (C = 128).
- A (cnn): full 1D ResNet, 5 blocks, kernel 7, RF 2.27 s, T' = 157. B (tcn): context-branch design at width 168 to match parameter counts.
- Params: A 1,235,237; B 1,255,709; C 1,192,005; D 1,240,773; E 1,273,669 (E − D = 32,896 = P plus the gate's extra input channels).
- Gate (D, E): channel-wise, per timestep, 2-layer MLP as 1×1 convs (in → 128 → C) + sigmoid, last bias initialised to 0 (r ≈ 0.5 at start). Reported r = channel mean. D and E differ only in the |F_local − P(F_ctx)| input (P = 1×1 conv C→C). F_local / F_ctx are taken after each branch's final ReLU.
