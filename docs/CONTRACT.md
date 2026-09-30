# Phase 1 interface contract (shared by all modules)

Binding spec: `docs/master.md` Part B, as reduced by B-scope (2026-09-29): 100 Hz, synthetic
corruptions only, variants A/C/E. This file fixes the cross-module interfaces
so modules written in parallel fit together. If you must deviate, keep the
signature and note it in your final report.

Environment: conda env `medico` (python 3.11, torch 2.13 w/ MPS, numpy 2.x,
scipy, pandas, scikit-learn, matplotlib, pyyaml, tqdm, pytest, wfdb 4.3).
Run everything as `conda run -n medico python ...` / `conda run -n medico pytest`.
Do NOT pip install anything new without saying so in the report.

## Layout
```
data.py  corruptions.py  models.py  train.py  eval.py
configs/phase1.yaml
tests/make_synthetic_data.py   # writes fake PTB-XL + NSTDB in PhysioNet formats
tests/test_*.py
data/ptb-xl/  data/nstdb/      # real data (read-only; may still be downloading)
data/cache/                    # derived caches (gitignored)
runs/  results/                # outputs (gitignored)
```

## Constants
```python
FS = 100                                  # Hz, Phase 1 default (records100); 500 = records500
N_SAMPLES = 1000                          # 10 s at FS; in general 10 * fs
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]   # label column order
```

## data.py
```python
def load_ptbxl(root: str, folds: list[int], lead: str = "II",
               cache_dir: str | None = "data/cache", fs: int = FS) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]
    # returns X_raw float32 [N, 10*fs] in mV (RAW, unfiltered),
    #         Y float32 [N, 5] multi-hot in SUPERCLASSES order,
    #         meta DataFrame with columns ecg_id, patient_id, strat_fold (row-aligned).
    # Labels: scp_codes keys that are diagnostic in scp_statements.csv
    # (diagnostic == 1) mapped via diagnostic_class; ALL codes regardless of
    # likelihood (as in the official PTB-XL example). Drop records with no superclass.
    # fs=100 -> filename_lr (records100), fs=500 -> filename_hr (records500).
    # Caches to cache_dir as .npz keyed by (fs, lead, folds).

def preprocess(x: np.ndarray, fs: int = FS) -> np.ndarray
    # x [..., T] -> band-pass 0.5-40 Hz (4th-order Butterworth, zero-phase
    # scipy.signal.sosfiltfilt along last axis), then per-record z-score
    # (eps 1e-6). float32 out. Called AFTER corruption.

def load_nstdb(root: str, target_fs: int = FS, train_frac: float = 0.6) -> dict[str, dict[str, np.ndarray]]
    # {"bw": {"train": arr, "test": arr}, "ma": {...}, "em": {...}}
    # arr float32 [T, 2] (both channels), resampled 360 -> target_fs
    # (scipy.signal.resample_poly). Split IN TIME: first 60% train, last 40% test.
    # Not used in Phase 1 (no NSTDB family configured).

class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, X_raw, Y, corruptor: "Corruptor | None" = None,
                 p_corrupt: float = 0.5, paired: bool = False, seed: int = 0, fs: int = FS)
    # __getitem__ returns dict:
    #   unpaired: {"x": float32 tensor [1, 10*fs] (preprocessed), "y": [5], "idx": int}
    #     with prob p_corrupt the raw signal is corrupted by corruptor.sample_train() first.
    #   paired:   {"x_clean": [1,T], "x_corr": [1,T], "y": [5], "idx": int,
    #              "severity": int (0=mild/15dB, 1=moderate/6dB, 2=severe/0dB)}
    #     x_corr ALWAYS corrupted with sample_train(); x_clean never corrupted.
    # RNG: per-worker np.random.Generator seeded from (seed, worker id, epoch-safe);
    #   provide `worker_init_fn(worker_id)` module-level helper.
```

## corruptions.py
```python
SEVERITY_SNR = {"mild": 15.0, "moderate": 6.0, "severe": 0.0}      # dB
DROPOUT_SEC  = {"mild": 0.5, "moderate": 1.0, "severe": 2.0}
SEVERITIES   = ["mild", "moderate", "severe"]                      # index = severity int
TRAIN_FAMILIES  = ["baseline_wander", "emg"]          # Phase 1 defaults; config: corruptions.*
UNSEEN_FAMILIES = ["motion_burst", "dropout"]
ALL_FAMILIES = TRAIN_FAMILIES + UNSEEN_FAMILIES
NSTDB_FAMILIES = ["nstdb_bw", "nstdb_ma", "nstdb_em"]   # later phases
KNOWN_FAMILIES = NSTDB_FAMILIES + ["baseline_wander", "emg", "powerline", "motion_burst", "dropout"]
MODES = ["whole", "burst"]            # burst = one contiguous 2-4 s segment

class Corruptor:
    def __init__(self, nstdb: dict | None, split: str, fs: int = FS,
                 train_families: list[str] | None = None)   # split in {"train","test"}
        # uses only nstdb[k][split] excerpts -> train/test noise disjoint.
        # nstdb=None -> nstdb_* families raise. powerline raises at fs <= 200.
        # train_families (default TRAIN_FAMILIES) drive sample_train; dropout not allowed.
    def apply(self, x: np.ndarray, family: str | list[str], severity: str,
              mode: str, rng: np.random.Generator) -> np.ndarray
        # x RAW [T] -> corrupted RAW [T]. family list => "mixed": sum the
        # unit-power noises of each family, then scale the SUM to the target SNR.
        # SNR = 10 log10(P_signal / P_noise), powers computed over the affected
        # segment only (whole record or the burst). Signal power on the
        # mean-removed segment. dropout: zero a segment of DROPOUT_SEC[severity]
        # (mode ignored). Random NSTDB excerpt start + random channel.
    def sample_train(self, x, rng) -> tuple[np.ndarray, int]
        # random family from self.train_families, or a mix of 2 of them with prob 0.25;
        # random severity (uniform); random mode (uniform). Returns (x_corr, severity_int).
```
Synthetic families: baseline_wander = sum of 1-3 sinusoids 0.05-0.5 Hz with random
phase; emg = band-limited Gaussian (20 Hz to min(150, 0.45*fs) Hz) noise; powerline = 50 Hz sinusoid
(+ small 2nd harmonic); motion_burst = low-frequency (1-10 Hz) high-amplitude
transients/steps (whole mode = several bursts, burst mode = one); dropout above.
All deterministic given rng.

## models.py
```python
MODEL_NAMES = ["cnn", "tcn", "concat", "gate", "racer"]
# Experiment variants (train.py maps these): A=cnn, B=tcn, C=concat, D=gate,
# E=racer, F=racer + rank/consistency losses, G=cnn + consistency loss.
def build_model(name: str, n_classes: int = 5, fs: int = 100, **kw) -> nn.Module
def count_params(model) -> int
# fs selects the stem strides / local depth (models.ARCH) so RFs in seconds match across rates.
# forward(x: [B, 1, 10*fs]) -> dict:
#   "logits": [B, n_classes]
#   "r":      [B, T'] per-timestep reliability in [0,1] (channel-mean of the gate)
#             for gate/racer; None for cnn/tcn/concat.
#   "r_map":  [B, C, T'] full gate (gate/racer) else None.
# All 5 models: 1-3M params. Two-branch models: local CNN branch and context
# TCN branch output identical [B, C, T'] maps; fusion per timestep; GAP; linear head.
```
Must work on device "mps" under `torch.autocast("mps", dtype=torch.float16)` and on CPU fp32.

## Losses (train.py)
- BCEWithLogits on logits. Paired mode: BCE averaged over x_clean and x_corr views
  (so marginal corruption prob = 0.5, matching regime ii).
- rank (F): per sample m = mean_t r; loss = mean(relu(m_corr - m_clean + margin)), margin=0.1, weight lambda2=0.1.
- consistency (F, G): only on pairs with severity==0 (mild, 15 dB):
  MSE between sigmoid(logits_corr) and sigmoid(logits_clean), gradient through
  both (symmetric), weight lambda1=0.1. Zero if no mild pairs in batch.

## CLI
```
python train.py --config configs/phase1.yaml --variant {A..G} --regime {clean,aug} --seed S [--smoke]
  # Phase 1 sweep (config `variants`): A, C, E x {clean, aug} x 5 seeds = 30 runs
  -> runs/{variant}_{regime}_s{seed}/ : best.pt, config.yaml (resolved), meta.json
     (git hash or "nogit", fs, param count, amp dtype used, device, epochs, best val AUROC),
     log.csv (epoch, train_loss, val_macro_auroc), thresholds.json (per-class F1-optimal on clean val).
python eval.py --config configs/phase1.yaml [--runs runs/] [--smoke]
  -> results/per_run.csv (variant, regime, seed, condition, family, severity, mode, macro_auroc, macro_f1)
     results/bootstrap.csv, results/reliability.csv, results/figures/*.png, summary.md
```
Variants F and G are only valid with regime aug.
