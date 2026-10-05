# Track 2 log

- 2026-10-01 00:47:28 RULES.md committed before any Track 2 run: commit `4c77715095033067a4537ef958d3c23a8c11c34c`.
- 2026-10-01 00:55:44 calibration done (fold-10 log #9-12); difficulty check done, severity set {-6 dB}; code for runs committed: `100e55535dd3f61fc004bd75c652653c86419944`.
- 2026-10-01 02:10:00 18 Track 2 runs trained; fold-9 grid (21 models x 49 conditions) evaluated; **fold-9 verdict NO-GO** (E fails rules 1-3, F fails rules 1-2). Fold 10 not scored for Track 2 variants (fold-10 log stays at 12).

## 2026-10-03 15:10 - RULES3.md written before any revision run
sha256 dfe1e316f2bbd15c0fc1bb5229d7212ed652bd42edd3a6695e12dd672c343887. Revision ablations (9 variants x 3 seeds,
fold 9 only, descriptive) and analyses of stored predictions. Code: t2_E_h128, t2_T (track2_models.py),
--aug-families (train.py, corruptions.random_seen). Smoke tests passed (1 epoch, 256 records).

## 2026-10-04 13:18 - RULES4.md written before any RULES4 run
sha256 f1fa7d34436dc562348b0226a88d137169bef426b135e12d8cf37c4852e7633a. Robustness suite: architectures (xresnet1d50, inception1d), wang seeds 3-4, Chapman-Shaoxing,
clamp sweep, selective prediction, Holm/TOST/meta-analysis. Fold 10 not touched.

## 2026-10-04 20:55 - RULES4.md overwritten by a preview report, restored
rules4_report.py wrote rules4.md, which on the case-insensitive macOS file system replaced RULES4.md (20:54). Restored
byte-identical (sha256 f1fa7d34... matches the entry above); report output renamed to rules4_report.{md,json}.
No decision depended on the file between those times.
