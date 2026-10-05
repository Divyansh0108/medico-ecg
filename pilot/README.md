# RACER — Phase 1 (reliability-aware ECG representation learning)

Phase 1 go/no-go experiment from `docs/master.md` Part B, reduced by the **B-scope decision of
2026-09-29**. The binding interfaces are in `docs/CONTRACT.md`.

**Phase 1 scope (what actually runs):**
- Data: PTB-XL only, the official 100 Hz files (`records100`), lead II, official `strat_fold`
  split (1–8 train, 9 val, 10 test), and 5 diagnostic superclasses (multilabel).
- Corruptions: synthetic only (no NSTDB). Training uses baseline wander and EMG. Motion bursts
  and dropout are unseen at test. Powerline cannot be represented at 100 Hz (Nyquist is 50 Hz).
- Models: A = CNN, C = CNN+TCN concat and E = RACER-lite, each in regime `clean` (no noise
  augmentation) and regime `aug` (with noise augmentation).

B = TCN, D = generic gate, F = RACER-lite + rank/consistency losses and G = CNN + consistency
loss stay implemented and tested, but they are not in the sweep (`variants:` in the config).
NSTDB and 500 Hz support are kept for later phases and switched on through the config.

## Setup

```
conda env create -f environment.yml     # creates env "medico"
conda activate medico                    # or prefix every command with: conda run -n medico
```

## Data

Phase 1 needs only PTB-XL (`ptbxl_database.csv`, `scp_statements.csv`, `records100/`). The
NSTDB records (`bw`, `ma`, `em`) and `records500/` are read only when the config asks for them.
Paths are set under `paths:` in `configs/phase1.yaml` (defaults: `../data/ptb-xl`, `../data/nstdb`).

```
# PTB-XL 1.0.3 (~3 GB)
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ ../data/ptb-xl/

# NSTDB 1.0.0, only the three noise records (not on the S3 mirror; use HTTPS)
mkdir -p ../data/nstdb && for f in bw ma em; do for e in dat hea; do
  curl -fL -o ../data/nstdb/$f.$e https://physionet.org/files/nstdb/1.0.0/$f.$e; done; done

# Phase 2 only
aws s3 cp --no-sign-request s3://physionet-open/challenge-2017/1.0.0/training2017.zip ../data/cinc2017/
aws s3 cp --no-sign-request s3://physionet-open/challenge-2017/1.0.0/REFERENCE-v3.csv ../data/cinc2017/
aws s3 sync --no-sign-request s3://physionet-open/butqdb/1.0.0/ ../data/butqdb/
```
`wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/` also works, but it nests the files
under `physionet.org/files/...`. If you use it, point `paths.ptbxl_root` at that folder.

## Commands

Unit tests:
```
conda run -n medico pytest tests -q
```

Smoke test (end to end on synthetic PhysioNet-format data; outputs go to `runs/smoke`, `results/smoke`):
```
conda run -n medico python tests/make_synthetic_data.py --out ../data/synthetic --n-records 200
scripts/run_all.sh --smoke                                   # 6 variant/regime pairs x 2 seeds = 12 runs
conda run -n medico python eval.py --config configs/phase1.yaml --smoke
```

Single run:
```
conda run -n medico python train.py --config configs/phase1.yaml --variant E --regime aug --seed 0
# -> runs/E_aug_s0/{best.pt, config.yaml, meta.json, log.csv, thresholds.json}
```

Full sweep: config `variants` × {clean, aug}, with paired F and G in `aug` only. Phase 1 is
A, C, E × {clean, aug} = 6 pairs; × 5 seeds = 30 runs, run one after another. The sweep can be
resumed: a run with both `best.pt` and `meta.json` is skipped.
```
scripts/run_all.sh
```

Evaluation (every complete run under `runs/`):
```
conda run -n medico python eval.py --config configs/phase1.yaml [--runs runs/]
```
Outputs in `results/`:
- `per_run.csv`: long format, one row per (variant, regime, seed, condition), with family,
  severity, mode, macro_auroc and macro_f1.
- `per_run_groups.csv`: group aggregates and drops, per run.
- `table_mean_std.csv` and `table_conditions_mean_std.csv`: mean ± std over seeds.
- `bootstrap.csv`
- `reliability.csv`
- `figures/auroc_vs_snr_{aug,clean}.png` and `figures/r_vs_snr.png`
- `summary.md`: the B8 verdict.

## Docs

- `docs/master.md`: research plan and Phase 1 spec (binding).
- `docs/actual.md`: research plan v2 with SOTA references.
- `docs/CONTRACT.md`: module interfaces.
- `docs/decisions.md`: implementation decisions for points the spec left open.
