# Track 2 log

- 2026-10-01 00:47:28 RULES.md committed before any Track 2 run: commit `4e0b3c2557239530843da0d6817379b6c8f8fd16`.
- 2026-10-01 00:55:44 calibration done (fold-10 log #9-12); difficulty check done, severity set {-6 dB}; code for runs committed: `100e55535dd3f61fc004bd75c652653c86419944`.
- 2026-10-01 02:10:00 18 Track 2 runs trained; fold-9 grid (21 models x 49 conditions) evaluated; **fold-9 verdict NO-GO** (E fails rules 1-3, F fails rules 1-2). Fold 10 not scored for Track 2 variants (fold-10 log stays at 12).

## 2026-10-03 15:10 - RULES3.md written before any revision run
sha256 dfe1e316f2bbd15c0fc1bb5229d7212ed652bd42edd3a6695e12dd672c343887. Revision ablations (9 variants x 3 seeds,
fold 9 only, descriptive) and analyses of stored predictions. Code: t2_E_h128, t2_T (track2_models.py),
--aug-families (train.py, corruptions.random_seen). Smoke tests passed (1 epoch, 256 records).
