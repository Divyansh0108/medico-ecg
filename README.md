# medico-ecg

**Realistic noise augmentation, not reliability gating, makes compact ECG classifiers robust.**
Code, pre-specified decision rules, logs and per-run results for a staged robustness study on PTB-XL and
Chapman-Shaoxing, with real-noise (MIT-BIH NSTDB) and external quality-label checks (CinC 2017, BUT QDB).

![python](https://img.shields.io/badge/python-3.11-blue) ![pytorch](https://img.shields.io/badge/PyTorch-MPS%2FCUDA%2FCPU-ee4c2c) ![tests](https://img.shields.io/badge/tests-280-brightgreen) ![runs](https://img.shields.io/badge/training%20runs-134-informational)

## Paper

> D. Pandey, S. Garg and V. Tiwari. *Realistic Noise Augmentation, Not Reliability Gating, Makes Compact ECG
> Classifiers Robust: A Pre-specified Study Across Architectures and Datasets.* Manuscript under review, 2026.

Divyansh Pandey and Shubh Garg contributed equally. Corresponding author: Varun Tiwari
(varun.tiwari@jaipur.manipal.edu).

```bibtex
@unpublished{pandey2026ecgrobust,
  author = {Pandey, Divyansh and Garg, Shubh and Tiwari, Varun},
  title  = {Realistic Noise Augmentation, Not Reliability Gating, Makes Compact {ECG} Classifiers Robust:
            A Pre-specified Study Across Architectures and Datasets},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## Findings

| Question | Answer |
|---|---|
| Can a compact supervised model reach benchmark accuracy on PTB-XL? | **Yes.** A 0.47M-parameter ResNet reaches 0.9315 ± 0.0010 fold-10 macro-AUROC over five seeds (published 0.930). Three seeds reach 0.9347 and a pre-specified five-model ensemble 0.9364, matching the published seven-model ensemble (0.934) with 4.5 and 1.3 times fewer parameters, 5.4 and 3.9 times fewer operations, and no pretraining. |
| What makes the model robust to noise? | **Augmentation.** Training with noise recovers +0.043 to +0.102 macro-AUROC on mixed noise at −6 dB. |
| Does a learned reliability gate add anything on top? | **No.** 0 of 8 Holm-corrected gate-vs-augmentation tests reach the pre-specified +0.005 bar; 6 of 8 are equivalent to zero (TOST, ±0.005). Pooled gain (DerSimonian-Laird): SevGate +0.0016 [+0.0001, +0.0031], DiffGate −0.0026. |
| What about real noise? | Both gates are **worse** than augmentation on held-out NSTDB noise (pooled −0.0098 and −0.0051). Putting real noise in training adds +0.019 to +0.059. |
| Is the gate input-dependent? | Barely. Replacing the learned gate by the best constant costs at most 0.0035 on synthetic noise and 0.0062 on NSTDB. |
| Is the gate score a useful quality signal? | It detects corruption (AUROC up to 1.00 with severity supervision), but abstaining by it is worse than abstaining by predictive entropy in 8 of 8 settings, and it does not behave like a quality index on CinC 2017 or BUT QDB. |

Settings of the robustness study: **S1** ResNet-Wang on PTB-XL (5 seeds), **S2** XResNet1d50 on PTB-XL,
**S3** Inception1d on PTB-XL, **S4** ResNet-Wang on Chapman-Shaoxing (3 seeds each). Full tables:
[results/robustness/rules4_report.md](benchmark/results/robustness/rules4_report.md).

### Model names

| paper | run name in code and results |
|---|---|
| Clean (no augmentation) | `B0-clean` |
| Aug (noise augmentation) | `B0-aug` |
| Concat | `C` |
| Gate (generic gate) | `D` |
| DiffGate (reliability gate) | `E` |
| SevGate (DiffGate + severity supervision) | `F` |
| DiffGate-clean / Aug-real / SevGate-real | `E-clean` / `B0-aug-real` / `F-real` |

## Pre-specification

The study ran in four stages. Each rules file was committed before the runs it governs; commit times are in
`git log`.

| stage | rules | runs | purpose |
|---|---|---|---|
| 1 | [RULES.md](benchmark/results/robustness/RULES.md) | 18 | primary GO/NO-GO verdict, PTB-XL, ResNet-Wang |
| 2 | [RULES2.md](benchmark/results/robustness/RULES2.md) | 24 | real-noise training, fold-10 reporting pass, single-lead models, CinC 2017, BUT QDB |
| 3 | [RULES3.md](benchmark/results/robustness/RULES3.md) | 27 | loss, width, temporal-resolution and augmentation-mix ablations |
| 4 | [RULES4.md](benchmark/results/robustness/RULES4.md) | 65 | other architectures and dataset, seeds, Holm/TOST/meta-analysis, clamp sweep, selective prediction |

- **Fold 9 for every choice.** PTB-XL fold 10 is scored only in logged passes:
  [FOLD10_LOG.md](benchmark/results/robustness/FOLD10_LOG.md).
- **Patient-level bootstrap** for all confidence intervals, paired for model differences.
- **Per-seed results** are reported next to ensembles.
- The rules files and logs were written before the repository was reorganized and use the old paths
  (`track1/`, `phase1/`, `results/track2/`, `results/crop/`, old module names). The mapping is in
  [benchmark/README.md](benchmark/README.md#old-names).

## Repository layout

```
.
├── benchmark/            12-lead PTB-XL and Chapman experiments (the paper)
│   ├── loaders/          PTB-XL and Chapman-Shaoxing loading, split and leakage checks
│   ├── noise/            synthetic artifact families and real NSTDB noise
│   ├── models/           compact baselines, PTB-XL benchmark architectures, gated variants
│   ├── training/         training loop, seed selection, seed ensembles
│   ├── evaluation/       metrics, robustness grids, calibration, fold-10 evaluation and log
│   ├── external/         single-lead inference, CinC 2017 and BUT QDB pipelines
│   ├── reporting/        report generators for every stage
│   ├── scripts/          resumable pipelines, one per stage
│   ├── tests/            85 tests
│   ├── results/          ptbxl_baselines/ and robustness/ (rules, reports, figures, logs of test-fold use)
│   ├── summaries/        plain-language summaries of each round
│   └── logs/             training and pipeline logs
├── pilot/                single-lead pilot (lead II, 30 runs, NO-GO); own README, configs and 195 tests
├── docs/                 project spec, research plan, interface contract, decision log
├── environment.yml       conda environment "medico"
└── data/                 datasets (not in git)
```

Not in git because of size: datasets (~8 GB), model checkpoints (`*.pt`) and cached predictions (`*.npz`).

## Reproduce

```bash
conda env create -f environment.yml && conda activate medico

# PTB-XL (PhysioNet, no login needed)
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ data/ptb-xl/
# NSTDB, CinC 2017 and BUT QDB: see pilot/README.md ("Data")
# Chapman-Shaoxing: the PhysioNet/CinC Challenge 2021 copy (WFDB .hea/.mat) in data/chapman/

cd benchmark
python -m loaders.check_splits                 # folds, prevalence, patient-leakage check
bash scripts/baselines_m1_m2.sh                # first baselines
bash scripts/ptbxl_seeds.sh && bash scripts/ptbxl_crops.sh            # PTB-XL benchmark and ensembles
bash scripts/rules1_train.sh 0 && bash scripts/rules1_eval.sh         # stage 1 (repeat training for seeds 1, 2)
bash scripts/rules2_real_noise_train.sh && bash scripts/rules2_single_lead_train.sh && bash scripts/rules2_external_eval.sh
bash scripts/rules3_ablation_train.sh && bash scripts/rules3_ablation_eval.sh
bash scripts/rules4_train.sh && bash scripts/rules4_eval.sh
pytest tests -q                                # 85 tests

cd ../pilot && pytest tests -q                 # 195 tests
```

Modules are run from `benchmark/` with `python -m <package>.<module>`. All scripts are resumable (completed
runs are skipped). Developed on Apple Silicon (MPS); fp16 was slower than fp32 there.

## Limitations

- PTB-XL at 100 Hz; the second dataset (Chapman-Shaoxing) is a rhythm task near ceiling on clean data.
- Scalar per-window gates only; no per-lead gates and no lead-localized artifacts.
- NSTDB is the only real-noise source, added to clinical recordings rather than recorded with them; the BUT QDB
  protocol is reconstructed from stated assumptions (RULES2.md).
- S2 to S4 use three seeds, which makes seed-level effect sizes imprecise.
- The negative result is about this gate design space, not reliability estimation in general.

## License

Code: [MIT](LICENSE). PTB-XL, Chapman-Shaoxing, NSTDB, CinC 2017 and BUT QDB are distributed by PhysioNet under
their own licenses and are not included.

## Documents

[Project spec](docs/master.md) · [Research plan](docs/actual.md) · [Interface contract](docs/CONTRACT.md) ·
[Decision log](docs/decisions.md) · [Stage 1 report](benchmark/results/robustness/REPORT.md) ·
[Stage 2 summary](benchmark/results/robustness/FOLLOWUPS_SUMMARY.md) ·
[Stage 4 report](benchmark/results/robustness/rules4_report.md)
