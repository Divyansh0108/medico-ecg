# medico — ECG reliability research (Q1 journal target)

| Folder | What | Status |
|---|---|---|
| `phase1/` | RACER Phase 1: PTB-XL lead II, synthetic noise, models A/C/E (30 runs). See `phase1/README.md`. | Done 2026-09-29, verdict NO-GO, paused |
| `track1/` | PTB-XL Track 1: 12-lead superclass benchmark, M1 (ResNet-SE) vs M2 (+transformer). Spec: `track1/ins.md`. Report: `track1/results/REPORT_s0.md`. | Seed 0 done 2026-09-29 (0.900 test mAUROC), awaiting decision |
| `docs/` | Project spec (`master.md`), Shubh's plan v2 (`actual.md`), interface contract, decisions log. | |
| `data/` | Shared datasets (PTB-XL, NSTDB, BUT QDB, CinC 2017) and caches. Not in git. | |

Setup: `conda env create -f environment.yml && conda activate medico`, then run scripts from inside
`phase1/` or `track1/` (both read the data from `../data/`).
