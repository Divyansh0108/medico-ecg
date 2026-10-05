# RULES4 report (robustness of the conclusion)

Ensembles: probability average over seeds (S1 wang: seeds 0-4 for Clean/Aug/Gate/DiffGate/SevGate). PTB-XL settings on fold 9; Chapman on its held-out test split. Groups at -6 dB, both modes.

## Confirmatory tests (mixed@-6, Holm over 8 one-sided tests, alpha 0.05; TOST margin 0.005)

| comparison | delta [95% CI] | 90% CI | one-sided p | Holm reject | verdict |
|---|---|---|---|---|---|
| DiffGate - Aug, wang | +0.0006 [-0.0013, +0.0026] | [-0.0010, +0.0022] | 0.270 | False | not significant; equivalent |
| SevGate - Aug, wang | +0.0024 [+0.0005, +0.0045] | [+0.0009, +0.0041] | 0.004 | True | significant, below practical bar; equivalent |
| DiffGate - Aug, xresnet | -0.0021 [-0.0041, +0.0000] | [-0.0038, -0.0003] | 0.974 | False | not significant; equivalent |
| SevGate - Aug, xresnet | +0.0026 [+0.0005, +0.0048] | [+0.0008, +0.0045] | 0.013 | False | not significant; equivalent |
| DiffGate - Aug, inception | -0.0041 [-0.0061, -0.0018] | [-0.0059, -0.0022] | 1.000 | False | not significant; not equivalent |
| SevGate - Aug, inception | -0.0006 [-0.0028, +0.0015] | [-0.0025, +0.0013] | 0.695 | False | not significant; equivalent |
| DiffGate - Aug, chapman | -0.0051 [-0.0079, -0.0026] | [-0.0074, -0.0031] | 1.000 | False | not significant; not equivalent |
| SevGate - Aug, chapman | +0.0018 [-0.0012, +0.0046] | [-0.0008, +0.0042] | 0.134 | False | not significant; equivalent |

## Random-effects meta-analysis over S1-S4 (DerSimonian-Laird)

| gate - Aug | endpoint | pooled [95% CI] | tau^2 | I^2 |
|---|---|---|---|---|
| DiffGate | mixed@-6 | -0.0026 [-0.0051, -0.0001] | 5.33e-06 | 0.81 |
| DiffGate | unseen@-6 | -0.0031 [-0.0058, -0.0003] | 7.01e-06 | 0.89 |
| DiffGate | nstdb_all@-6 | -0.0098 [-0.0139, -0.0058] | 1.60e-05 | 0.93 |
| SevGate | mixed@-6 | +0.0016 [+0.0001, +0.0031] | 1.07e-06 | 0.44 |
| SevGate | unseen@-6 | +0.0013 [-0.0007, +0.0033] | 3.35e-06 | 0.79 |
| SevGate | nstdb_all@-6 | -0.0051 [-0.0087, -0.0014] | 1.24e-05 | 0.90 |

## Setting wang (N = 2146)

| model | clean | seen | unseen | mixed | all corr. | NSTDB all | bw | emg | motion | dropout | powerline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clean | 0.9394 | 0.8932 | 0.9005 | 0.8669 | 0.8925 | 0.8666 | 0.8952 | 0.8912 | 0.9070 | 0.8871 | 0.9074 |
| Aug | 0.9387 | 0.9240 | 0.9161 | 0.9101 | 0.9177 | 0.8897 | 0.9218 | 0.9261 | 0.9178 | 0.9020 | 0.9283 |
| Gate | 0.9381 | 0.9236 | 0.9127 | 0.9086 | 0.9157 | 0.8760 | 0.9225 | 0.9248 | 0.9144 | 0.8956 | 0.9281 |
| DiffGate | 0.9386 | 0.9250 | 0.9157 | 0.9107 | 0.9179 | 0.8750 | 0.9240 | 0.9261 | 0.9170 | 0.9013 | 0.9286 |
| SevGate | 0.9380 | 0.9269 | 0.9179 | 0.9125 | 0.9200 | 0.8811 | 0.9263 | 0.9274 | 0.9184 | 0.9057 | 0.9295 |
| DiffGate-clean | 0.9376 | 0.8604 | 0.8789 | 0.8438 | 0.8669 | 0.8371 | 0.8765 | 0.8443 | 0.8951 | 0.8615 | 0.8801 |
| Clean (s0-2) | 0.9389 | 0.8947 | 0.8985 | 0.8661 | 0.8919 | 0.8649 | 0.8957 | 0.8938 | 0.9054 | 0.8847 | 0.9055 |
| Aug (s0-2) | 0.9389 | 0.9241 | 0.9161 | 0.9094 | 0.9177 | 0.8895 | 0.9219 | 0.9263 | 0.9180 | 0.9015 | 0.9289 |
| DiffGate (s0-2) | 0.9381 | 0.9231 | 0.9148 | 0.9097 | 0.9167 | 0.8735 | 0.9222 | 0.9239 | 0.9166 | 0.9016 | 0.9263 |
| SevGate (s0-2) | 0.9380 | 0.9257 | 0.9173 | 0.9115 | 0.9191 | 0.8799 | 0.9251 | 0.9263 | 0.9183 | 0.9054 | 0.9282 |
| Aug-real | - | - | - | - | - | 0.9150 | - | - | - | - | - |
| SevGate-real | - | - | - | - | - | 0.9145 | - | - | - | - | - |

Per seed (mean +- s.d.): Clean mixed@-6 0.8560 +- 0.0042; Aug mixed@-6 0.9037 +- 0.0037; Gate mixed@-6 0.8990 +- 0.0034; DiffGate mixed@-6 0.9000 +- 0.0030; SevGate mixed@-6 0.9042 +- 0.0014; DiffGate-clean mixed@-6 0.8286 +- 0.0147; Clean (s0-2) mixed@-6 0.8570 +- 0.0034; Aug (s0-2) mixed@-6 0.9042 +- 0.0050; DiffGate (s0-2) mixed@-6 0.9000 +- 0.0030; SevGate (s0-2) mixed@-6 0.9041 +- 0.0008

| comparison | endpoint | delta [95% CI] | p (one-sided) | TOST equiv. | d_z (seeds) |
|---|---|---|---|---|---|
| DiffGate - Aug | mixed@-6 | +0.0006 [-0.0013, +0.0026] | 0.270 | True | -0.65 (n=5) |
| DiffGate - Aug | unseen@-6 | -0.0004 [-0.0019, +0.0013] | 0.670 | True | -0.99 (n=5) |
| DiffGate - Aug | clean | -0.0001 [-0.0011, +0.0011] | 0.552 | True | -0.88 (n=5) |
| DiffGate - Aug | nstdb_all@-6 | -0.0146 [-0.0166, -0.0127] | 1.000 | False | -3.11 (n=5) |
| SevGate - Aug | mixed@-6 | +0.0024 [+0.0005, +0.0045] | 0.004 | True | +0.14 (n=5) |
| SevGate - Aug | unseen@-6 | +0.0018 [+0.0002, +0.0037] | 0.015 | True | -0.17 (n=5) |
| SevGate - Aug | clean | -0.0007 [-0.0019, +0.0007] | 0.842 | True | -1.72 (n=5) |
| SevGate - Aug | nstdb_all@-6 | -0.0085 [-0.0103, -0.0067] | 1.000 | False | -1.68 (n=5) |
| Gate - Aug | mixed@-6 | -0.0015 [-0.0033, +0.0002] | 0.958 | True | -1.23 (n=5) |
| Gate - Aug | unseen@-6 | -0.0034 [-0.0050, -0.0018] | 1.000 | True | -2.04 (n=5) |
| Gate - Aug | clean | -0.0006 [-0.0019, +0.0007] | 0.798 | True | -2.68 (n=5) |
| SevGate-real - Aug-real | nstdb_all@-6 | -0.0006 [-0.0021, +0.0009] | 0.776 | True | -0.21 (n=3) |

Share of the augmentation gain added by the gate: DiffGate mixed@-6 +0.01; DiffGate unseen@-6 -0.03; SevGate mixed@-6 +0.06; SevGate unseen@-6 +0.12

r diagnostics: Gate: clean 0.382, -6 dB 0.330, Spearman(SNR, r) +0.174; DiffGate: clean 0.356, -6 dB 0.305, Spearman(SNR, r) +0.180; SevGate: clean 0.335, -6 dB 0.174, Spearman(SNR, r) +0.411; DiffGate-clean: clean 0.372, -6 dB 0.411, Spearman(SNR, r) -0.112; DiffGate (s0-2): clean 0.323, -6 dB 0.270, Spearman(SNR, r) +0.185; SevGate (s0-2): clean 0.317, -6 dB 0.167, Spearman(SNR, r) +0.392

Clamp sweep (seed-averaged; r = 1 uses F_local only):

| model | r | clean | mixed@-6 | unseen@-6 | NSTDB all@-6 |
|---|---|---|---|---|---|
| DiffGate | learned | 0.9386 | 0.9107 | 0.9157 | 0.8750 |
| DiffGate | r=0 | 0.9363 | 0.9069 | 0.9124 | 0.8689 |
| DiffGate | r=0.25 | 0.9366 | 0.9079 | 0.9137 | 0.8686 |
| DiffGate | r=0.5 | 0.9355 | 0.9044 | 0.9104 | 0.8638 |
| DiffGate | r=0.75 | 0.9268 | 0.8779 | 0.8821 | 0.8359 |
| DiffGate | r=1 | 0.6524 | 0.5467 | 0.5534 | 0.5295 |
| DiffGate | learned - best const. | +0.0020 | +0.0027 | +0.0019 | +0.0062 |

DiffGate: a constant r within 0.002 of learned on every condition: False

| SevGate | learned | 0.9380 | 0.9125 | 0.9179 | 0.8811 |
| SevGate | r=0 | 0.9360 | 0.9112 | 0.9167 | 0.8777 |
| SevGate | r=0.25 | 0.9364 | 0.9111 | 0.9168 | 0.8765 |
| SevGate | r=0.5 | 0.9360 | 0.9074 | 0.9133 | 0.8712 |
| SevGate | r=0.75 | 0.9315 | 0.8851 | 0.8909 | 0.8498 |
| SevGate | r=1 | 0.7571 | 0.6181 | 0.6224 | 0.6056 |
| SevGate | learned - best const. | +0.0016 | +0.0013 | +0.0011 | +0.0035 |

SevGate: a constant r within 0.002 of learned on every condition: False


Selective prediction, 50/50 clean / synthetic mixed|-6|whole:

| model | score | detect AUROC | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|---|---|
| Aug | entropy | 0.907 | 0.9094 | 0.9181 | 0.9280 | 0.9367 | 0.9439 | 0.9523 |
| Aug | H1 | 0.481 | 0.9094 | 0.9099 | 0.9141 | 0.9163 | 0.9148 | 0.9136 |
| Aug | random | - | 0.9094 | 0.9097 | 0.9094 | 0.9094 | 0.9092 | 0.9089 |
| DiffGate | entropy | 0.911 | 0.9105 | 0.9196 | 0.9286 | 0.9375 | 0.9449 | 0.9508 |
| DiffGate | H1 | 0.481 | 0.9105 | 0.9103 | 0.9140 | 0.9156 | 0.9140 | 0.9131 |
| DiffGate | -r | 0.793 | 0.9105 | 0.9121 | 0.9132 | 0.9135 | 0.9129 | 0.9153 |
| DiffGate | random | - | 0.9105 | 0.9108 | 0.9106 | 0.9106 | 0.9105 | 0.9099 |
| SevGate | entropy | 0.885 | 0.9113 | 0.9197 | 0.9292 | 0.9375 | 0.9460 | 0.9520 |
| SevGate | H1 | 0.481 | 0.9113 | 0.9108 | 0.9144 | 0.9161 | 0.9147 | 0.9138 |
| SevGate | -r | 0.985 | 0.9113 | 0.9113 | 0.9145 | 0.9205 | 0.9301 | 0.9378 |
| SevGate | random | - | 0.9113 | 0.9115 | 0.9113 | 0.9111 | 0.9112 | 0.9105 |

Selective prediction, 50/50 clean / nstdb_mixed|-6|whole:

| model | score | detect AUROC | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|---|---|
| Aug | entropy | 0.946 | 0.8853 | 0.8921 | 0.9023 | 0.9185 | 0.9330 | 0.9449 |
| Aug | H1 | 0.662 | 0.8853 | 0.8912 | 0.8971 | 0.9022 | 0.9049 | 0.9053 |
| Aug | random | - | 0.8853 | 0.8853 | 0.8853 | 0.8857 | 0.8860 | 0.8838 |
| Aug-real | entropy | 0.876 | 0.9157 | 0.9263 | 0.9339 | 0.9391 | 0.9458 | 0.9542 |
| Aug-real | H1 | 0.662 | 0.9157 | 0.9172 | 0.9200 | 0.9228 | 0.9248 | 0.9259 |
| Aug-real | random | - | 0.9157 | 0.9156 | 0.9153 | 0.9160 | 0.9166 | 0.9145 |
| DiffGate | entropy | 0.931 | 0.8739 | 0.8796 | 0.8907 | 0.9081 | 0.9270 | 0.9408 |
| DiffGate | H1 | 0.662 | 0.8739 | 0.8825 | 0.8909 | 0.8979 | 0.9004 | 0.8997 |
| DiffGate | -r | 0.146 | 0.8739 | 0.8648 | 0.8558 | 0.8433 | 0.8407 | 0.8348 |
| DiffGate | random | - | 0.8739 | 0.8739 | 0.8740 | 0.8743 | 0.8747 | 0.8722 |
| SevGate | entropy | 0.916 | 0.8776 | 0.8838 | 0.8958 | 0.9119 | 0.9300 | 0.9426 |
| SevGate | H1 | 0.662 | 0.8776 | 0.8852 | 0.8928 | 0.8990 | 0.9011 | 0.9008 |
| SevGate | -r | 0.342 | 0.8776 | 0.8736 | 0.8682 | 0.8614 | 0.8561 | 0.8563 |
| SevGate | random | - | 0.8776 | 0.8776 | 0.8776 | 0.8779 | 0.8782 | 0.8756 |
| SevGate-real | entropy | 0.795 | 0.9179 | 0.9283 | 0.9361 | 0.9418 | 0.9479 | 0.9540 |
| SevGate-real | H1 | 0.662 | 0.9179 | 0.9190 | 0.9215 | 0.9238 | 0.9246 | 0.9255 |
| SevGate-real | -r | 0.971 | 0.9179 | 0.9186 | 0.9205 | 0.9246 | 0.9299 | 0.9350 |
| SevGate-real | random | - | 0.9179 | 0.9178 | 0.9176 | 0.9180 | 0.9184 | 0.9171 |

## Setting xresnet (N = 2146)

| model | clean | seen | unseen | mixed | all corr. | NSTDB all | bw | emg | motion | dropout | powerline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clean | 0.9371 | 0.8623 | 0.8570 | 0.8046 | 0.8500 | 0.8423 | 0.8744 | 0.8502 | 0.8875 | 0.8381 | 0.8454 |
| Aug | 0.9372 | 0.9223 | 0.9137 | 0.9055 | 0.9152 | 0.8961 | 0.9226 | 0.9221 | 0.9199 | 0.9025 | 0.9186 |
| DiffGate | 0.9364 | 0.9225 | 0.9122 | 0.9035 | 0.9142 | 0.8887 | 0.9226 | 0.9225 | 0.9174 | 0.8997 | 0.9196 |
| SevGate | 0.9349 | 0.9249 | 0.9174 | 0.9082 | 0.9184 | 0.8928 | 0.9259 | 0.9240 | 0.9197 | 0.9071 | 0.9254 |
| DiffGate-clean | 0.9365 | 0.8768 | 0.8719 | 0.8293 | 0.8664 | 0.8547 | 0.8885 | 0.8652 | 0.8916 | 0.8662 | 0.8578 |
| Aug-real | 0.9369 | 0.9191 | 0.9114 | 0.9052 | 0.9129 | 0.9148 | 0.9201 | 0.9180 | 0.9187 | 0.9030 | 0.9126 |
| SevGate-real | 0.9357 | 0.9224 | 0.9159 | 0.9089 | 0.9169 | 0.9164 | 0.9238 | 0.9211 | 0.9194 | 0.9080 | 0.9202 |

Per seed (mean +- s.d.): Clean mixed@-6 0.7878 +- 0.0293; Aug mixed@-6 0.8955 +- 0.0010; DiffGate mixed@-6 0.8925 +- 0.0044; SevGate mixed@-6 0.9006 +- 0.0031; DiffGate-clean mixed@-6 0.8071 +- 0.0193; Aug-real mixed@-6 0.8923 +- 0.0008; SevGate-real mixed@-6 0.9002 +- 0.0014

| comparison | endpoint | delta [95% CI] | p (one-sided) | TOST equiv. | d_z (seeds) |
|---|---|---|---|---|---|
| DiffGate - Aug | mixed@-6 | -0.0021 [-0.0041, +0.0000] | 0.974 | True | -0.86 (n=3) |
| DiffGate - Aug | unseen@-6 | -0.0014 [-0.0035, +0.0004] | 0.929 | True | -2.59 (n=3) |
| DiffGate - Aug | clean | -0.0007 [-0.0023, +0.0006] | 0.850 | True | -20.32 (n=3) |
| DiffGate - Aug | nstdb_all@-6 | -0.0074 [-0.0096, -0.0052] | 1.000 | False | -2.65 (n=3) |
| SevGate - Aug | mixed@-6 | +0.0026 [+0.0005, +0.0048] | 0.013 | True | +1.52 (n=3) |
| SevGate - Aug | unseen@-6 | +0.0037 [+0.0020, +0.0056] | 0.000 | False | +3.21 (n=3) |
| SevGate - Aug | clean | -0.0023 [-0.0038, -0.0007] | 0.998 | True | -1.62 (n=3) |
| SevGate - Aug | nstdb_all@-6 | -0.0033 [-0.0057, -0.0008] | 0.998 | False | -0.31 (n=3) |
| SevGate-real - Aug-real | nstdb_all@-6 | +0.0016 [+0.0002, +0.0029] | 0.006 | True | +1.93 (n=3) |

Share of the augmentation gain added by the gate: DiffGate mixed@-6 -0.02; DiffGate unseen@-6 -0.03; SevGate mixed@-6 +0.03; SevGate unseen@-6 +0.07

r diagnostics: DiffGate: clean 0.170, -6 dB 0.167, Spearman(SNR, r) +0.020; SevGate: clean 0.301, -6 dB 0.143, Spearman(SNR, r) +0.620; DiffGate-clean: clean 0.237, -6 dB 0.222, Spearman(SNR, r) +0.188; SevGate-real: clean 0.291, -6 dB 0.134, Spearman(SNR, r) +0.613

Clamp sweep (seed-averaged; r = 1 uses F_local only):

| model | r | clean | mixed@-6 | unseen@-6 | NSTDB all@-6 |
|---|---|---|---|---|---|
| DiffGate | learned | 0.9364 | 0.9035 | 0.9122 | 0.8887 |
| DiffGate | r=0 | 0.9364 | 0.9039 | 0.9123 | 0.8883 |
| DiffGate | r=0.25 | 0.9365 | 0.9029 | 0.9116 | 0.8876 |
| DiffGate | r=0.5 | 0.9358 | 0.8993 | 0.9088 | 0.8840 |
| DiffGate | r=0.75 | 0.9315 | 0.8844 | 0.8961 | 0.8703 |
| DiffGate | r=1 | 0.8620 | 0.7670 | 0.7844 | 0.7784 |
| DiffGate | learned - best const. | -0.0001 | -0.0004 | -0.0001 | +0.0004 |

DiffGate: a constant r within 0.002 of learned on every condition: True

| SevGate | learned | 0.9349 | 0.9082 | 0.9174 | 0.8928 |
| SevGate | r=0 | 0.9349 | 0.9082 | 0.9173 | 0.8926 |
| SevGate | r=0.25 | 0.9349 | 0.9082 | 0.9175 | 0.8933 |
| SevGate | r=0.5 | 0.9346 | 0.9069 | 0.9166 | 0.8919 |
| SevGate | r=0.75 | 0.9329 | 0.8995 | 0.9100 | 0.8826 |
| SevGate | r=1 | 0.9083 | 0.8438 | 0.8561 | 0.8294 |
| SevGate | learned - best const. | -0.0000 | -0.0000 | -0.0001 | -0.0005 |

SevGate: a constant r within 0.002 of learned on every condition: True


Selective prediction, 50/50 clean / synthetic mixed|-6|whole:

| model | score | detect AUROC | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|---|---|
| Aug | entropy | 0.950 | 0.9032 | 0.9106 | 0.9196 | 0.9290 | 0.9369 | 0.9450 |
| Aug | H1 | 0.481 | 0.9032 | 0.9048 | 0.9087 | 0.9102 | 0.9095 | 0.9077 |
| Aug | random | - | 0.9032 | 0.9034 | 0.9034 | 0.9033 | 0.9028 | 0.9027 |
| Aug-real | entropy | 0.961 | 0.9044 | 0.9142 | 0.9228 | 0.9300 | 0.9374 | 0.9449 |
| Aug-real | H1 | 0.481 | 0.9044 | 0.9069 | 0.9101 | 0.9121 | 0.9116 | 0.9105 |
| Aug-real | random | - | 0.9044 | 0.9047 | 0.9047 | 0.9046 | 0.9041 | 0.9038 |
| DiffGate | entropy | 0.953 | 0.9032 | 0.9125 | 0.9213 | 0.9305 | 0.9375 | 0.9456 |
| DiffGate | H1 | 0.481 | 0.9032 | 0.9049 | 0.9086 | 0.9099 | 0.9082 | 0.9070 |
| DiffGate | -r | 0.522 | 0.9032 | 0.8959 | 0.8865 | 0.8837 | 0.8843 | 0.8789 |
| DiffGate | random | - | 0.9032 | 0.9036 | 0.9031 | 0.9032 | 0.9029 | 0.9025 |
| SevGate | entropy | 0.921 | 0.9071 | 0.9165 | 0.9256 | 0.9332 | 0.9403 | 0.9469 |
| SevGate | H1 | 0.481 | 0.9071 | 0.9081 | 0.9113 | 0.9134 | 0.9133 | 0.9134 |
| SevGate | -r | 1.000 | 0.9071 | 0.9100 | 0.9131 | 0.9192 | 0.9271 | 0.9340 |
| SevGate | random | - | 0.9071 | 0.9073 | 0.9073 | 0.9069 | 0.9070 | 0.9065 |
| SevGate-real | entropy | 0.935 | 0.9082 | 0.9178 | 0.9253 | 0.9336 | 0.9397 | 0.9464 |
| SevGate-real | H1 | 0.481 | 0.9082 | 0.9095 | 0.9126 | 0.9143 | 0.9141 | 0.9147 |
| SevGate-real | -r | 1.000 | 0.9082 | 0.9096 | 0.9147 | 0.9214 | 0.9269 | 0.9344 |
| SevGate-real | random | - | 0.9082 | 0.9085 | 0.9085 | 0.9082 | 0.9080 | 0.9077 |

Selective prediction, 50/50 clean / nstdb_mixed|-6|whole:

| model | score | detect AUROC | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|---|---|
| Aug | entropy | 0.942 | 0.8892 | 0.8927 | 0.9018 | 0.9149 | 0.9295 | 0.9434 |
| Aug | H1 | 0.662 | 0.8892 | 0.8934 | 0.8980 | 0.9028 | 0.9053 | 0.9072 |
| Aug | random | - | 0.8892 | 0.8891 | 0.8891 | 0.8899 | 0.8897 | 0.8882 |
| Aug-real | entropy | 0.928 | 0.9145 | 0.9233 | 0.9308 | 0.9368 | 0.9421 | 0.9498 |
| Aug-real | H1 | 0.662 | 0.9145 | 0.9165 | 0.9191 | 0.9223 | 0.9245 | 0.9269 |
| Aug-real | random | - | 0.9145 | 0.9145 | 0.9143 | 0.9150 | 0.9149 | 0.9134 |
| DiffGate | entropy | 0.964 | 0.8857 | 0.8938 | 0.9018 | 0.9149 | 0.9305 | 0.9416 |
| DiffGate | H1 | 0.662 | 0.8857 | 0.8911 | 0.8962 | 0.9012 | 0.9031 | 0.9041 |
| DiffGate | -r | 0.271 | 0.8857 | 0.8739 | 0.8635 | 0.8524 | 0.8426 | 0.8388 |
| DiffGate | random | - | 0.8857 | 0.8858 | 0.8853 | 0.8864 | 0.8861 | 0.8842 |
| SevGate | entropy | 0.953 | 0.8848 | 0.8928 | 0.9002 | 0.9135 | 0.9291 | 0.9400 |
| SevGate | H1 | 0.662 | 0.8848 | 0.8896 | 0.8941 | 0.8983 | 0.9005 | 0.9018 |
| SevGate | -r | 0.998 | 0.8848 | 0.8895 | 0.8936 | 0.9009 | 0.9130 | 0.9315 |
| SevGate | random | - | 0.8848 | 0.8848 | 0.8846 | 0.8851 | 0.8852 | 0.8833 |
| SevGate-real | entropy | 0.869 | 0.9156 | 0.9249 | 0.9314 | 0.9384 | 0.9436 | 0.9503 |
| SevGate-real | H1 | 0.662 | 0.9156 | 0.9168 | 0.9195 | 0.9220 | 0.9238 | 0.9260 |
| SevGate-real | -r | 1.000 | 0.9156 | 0.9175 | 0.9185 | 0.9228 | 0.9279 | 0.9344 |
| SevGate-real | random | - | 0.9156 | 0.9156 | 0.9154 | 0.9158 | 0.9159 | 0.9147 |

## Setting inception (N = 2146)

| model | clean | seen | unseen | mixed | all corr. | NSTDB all | bw | emg | motion | dropout | powerline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clean | 0.9384 | 0.8774 | 0.8826 | 0.8507 | 0.8755 | 0.8631 | 0.8934 | 0.8613 | 0.9062 | 0.8796 | 0.8619 |
| Aug | 0.9387 | 0.9213 | 0.9144 | 0.9068 | 0.9155 | 0.8923 | 0.9228 | 0.9199 | 0.9213 | 0.9017 | 0.9203 |
| DiffGate | 0.9380 | 0.9210 | 0.9081 | 0.9028 | 0.9115 | 0.8803 | 0.9199 | 0.9221 | 0.9180 | 0.8928 | 0.9134 |
| SevGate | 0.9373 | 0.9235 | 0.9133 | 0.9062 | 0.9155 | 0.8845 | 0.9237 | 0.9232 | 0.9202 | 0.8987 | 0.9209 |
| DiffGate-clean | 0.9389 | 0.8338 | 0.8352 | 0.8052 | 0.8297 | 0.8117 | 0.8369 | 0.8307 | 0.8879 | 0.8384 | 0.7794 |
| Aug-real | 0.9397 | 0.9213 | 0.9168 | 0.9090 | 0.9170 | 0.9176 | 0.9219 | 0.9207 | 0.9220 | 0.9058 | 0.9225 |
| SevGate-real | 0.9367 | 0.9208 | 0.9143 | 0.9074 | 0.9153 | 0.9129 | 0.9209 | 0.9206 | 0.9202 | 0.9028 | 0.9200 |

Per seed (mean +- s.d.): Clean mixed@-6 0.8381 +- 0.0081; Aug mixed@-6 0.8985 +- 0.0034; DiffGate mixed@-6 0.8916 +- 0.0049; SevGate mixed@-6 0.8971 +- 0.0048; DiffGate-clean mixed@-6 0.7849 +- 0.0030; Aug-real mixed@-6 0.9008 +- 0.0033; SevGate-real mixed@-6 0.8975 +- 0.0047

| comparison | endpoint | delta [95% CI] | p (one-sided) | TOST equiv. | d_z (seeds) |
|---|---|---|---|---|---|
| DiffGate - Aug | mixed@-6 | -0.0041 [-0.0061, -0.0018] | 1.000 | False | -4.07 (n=3) |
| DiffGate - Aug | unseen@-6 | -0.0064 [-0.0082, -0.0047] | 1.000 | False | -5.48 (n=3) |
| DiffGate - Aug | clean | -0.0007 [-0.0022, +0.0007] | 0.839 | True | -2.06 (n=3) |
| DiffGate - Aug | nstdb_all@-6 | -0.0119 [-0.0141, -0.0099] | 1.000 | False | -1.89 (n=3) |
| SevGate - Aug | mixed@-6 | -0.0006 [-0.0028, +0.0015] | 0.695 | True | -0.60 (n=3) |
| SevGate - Aug | unseen@-6 | -0.0012 [-0.0030, +0.0006] | 0.901 | True | -1.65 (n=3) |
| SevGate - Aug | clean | -0.0014 [-0.0030, +0.0001] | 0.966 | True | -5.85 (n=3) |
| SevGate - Aug | nstdb_all@-6 | -0.0078 [-0.0101, -0.0056] | 1.000 | False | -1.01 (n=3) |
| SevGate-real - Aug-real | nstdb_all@-6 | -0.0047 [-0.0062, -0.0032] | 1.000 | False | -4.15 (n=3) |

Share of the augmentation gain added by the gate: DiffGate mixed@-6 -0.07; DiffGate unseen@-6 -0.20; SevGate mixed@-6 -0.01; SevGate unseen@-6 -0.04

r diagnostics: DiffGate: clean 0.276, -6 dB 0.172, Spearman(SNR, r) +0.414; SevGate: clean 0.357, -6 dB 0.143, Spearman(SNR, r) +0.573; DiffGate-clean: clean 0.397, -6 dB 0.279, Spearman(SNR, r) +0.456; SevGate-real: clean 0.373, -6 dB 0.148, Spearman(SNR, r) +0.577

Clamp sweep (seed-averaged; r = 1 uses F_local only):

| model | r | clean | mixed@-6 | unseen@-6 | NSTDB all@-6 |
|---|---|---|---|---|---|
| DiffGate | learned | 0.9380 | 0.9028 | 0.9081 | 0.8803 |
| DiffGate | r=0 | 0.9363 | 0.9000 | 0.9053 | 0.8750 |
| DiffGate | r=0.25 | 0.9365 | 0.8983 | 0.9037 | 0.8723 |
| DiffGate | r=0.5 | 0.9355 | 0.8906 | 0.8951 | 0.8615 |
| DiffGate | r=0.75 | 0.9282 | 0.8589 | 0.8607 | 0.8299 |
| DiffGate | r=1 | 0.5924 | 0.5505 | 0.5540 | 0.5416 |
| DiffGate | learned - best const. | +0.0015 | +0.0028 | +0.0027 | +0.0054 |

DiffGate: a constant r within 0.002 of learned on every condition: False

| SevGate | learned | 0.9373 | 0.9062 | 0.9133 | 0.8845 |
| SevGate | r=0 | 0.9358 | 0.9052 | 0.9122 | 0.8805 |
| SevGate | r=0.25 | 0.9361 | 0.9038 | 0.9103 | 0.8776 |
| SevGate | r=0.5 | 0.9360 | 0.8982 | 0.9035 | 0.8687 |
| SevGate | r=0.75 | 0.9327 | 0.8737 | 0.8759 | 0.8418 |
| SevGate | r=1 | 0.7640 | 0.6464 | 0.6479 | 0.6204 |
| SevGate | learned - best const. | +0.0012 | +0.0010 | +0.0011 | +0.0040 |

SevGate: a constant r within 0.002 of learned on every condition: False


Selective prediction, 50/50 clean / synthetic mixed|-6|whole:

| model | score | detect AUROC | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|---|---|
| Aug | entropy | 0.916 | 0.9066 | 0.9159 | 0.9246 | 0.9331 | 0.9427 | 0.9485 |
| Aug | H1 | 0.481 | 0.9066 | 0.9078 | 0.9124 | 0.9132 | 0.9125 | 0.9110 |
| Aug | random | - | 0.9066 | 0.9069 | 0.9068 | 0.9067 | 0.9062 | 0.9060 |
| Aug-real | entropy | 0.917 | 0.9100 | 0.9201 | 0.9297 | 0.9359 | 0.9435 | 0.9500 |
| Aug-real | H1 | 0.481 | 0.9100 | 0.9116 | 0.9158 | 0.9167 | 0.9164 | 0.9147 |
| Aug-real | random | - | 0.9100 | 0.9102 | 0.9104 | 0.9100 | 0.9097 | 0.9097 |
| DiffGate | entropy | 0.946 | 0.9015 | 0.9089 | 0.9191 | 0.9286 | 0.9376 | 0.9457 |
| DiffGate | H1 | 0.481 | 0.9015 | 0.9018 | 0.9070 | 0.9075 | 0.9064 | 0.9029 |
| DiffGate | -r | 0.969 | 0.9015 | 0.9051 | 0.9082 | 0.9148 | 0.9220 | 0.9267 |
| DiffGate | random | - | 0.9015 | 0.9018 | 0.9016 | 0.9015 | 0.9013 | 0.9012 |
| SevGate | entropy | 0.902 | 0.9040 | 0.9127 | 0.9224 | 0.9324 | 0.9424 | 0.9496 |
| SevGate | H1 | 0.481 | 0.9040 | 0.9038 | 0.9081 | 0.9089 | 0.9085 | 0.9061 |
| SevGate | -r | 1.000 | 0.9040 | 0.9043 | 0.9089 | 0.9152 | 0.9251 | 0.9354 |
| SevGate | random | - | 0.9040 | 0.9041 | 0.9040 | 0.9038 | 0.9037 | 0.9037 |
| SevGate-real | entropy | 0.940 | 0.9075 | 0.9173 | 0.9254 | 0.9322 | 0.9411 | 0.9485 |
| SevGate-real | H1 | 0.481 | 0.9075 | 0.9078 | 0.9119 | 0.9129 | 0.9130 | 0.9116 |
| SevGate-real | -r | 1.000 | 0.9075 | 0.9102 | 0.9141 | 0.9173 | 0.9259 | 0.9362 |
| SevGate-real | random | - | 0.9075 | 0.9076 | 0.9077 | 0.9074 | 0.9074 | 0.9073 |

Selective prediction, 50/50 clean / nstdb_mixed|-6|whole:

| model | score | detect AUROC | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|---|---|
| Aug | entropy | 0.931 | 0.8875 | 0.8914 | 0.9000 | 0.9145 | 0.9315 | 0.9424 |
| Aug | H1 | 0.662 | 0.8875 | 0.8923 | 0.8983 | 0.9034 | 0.9051 | 0.9062 |
| Aug | random | - | 0.8875 | 0.8876 | 0.8873 | 0.8880 | 0.8878 | 0.8866 |
| Aug-real | entropy | 0.877 | 0.9184 | 0.9286 | 0.9356 | 0.9416 | 0.9486 | 0.9526 |
| Aug-real | H1 | 0.662 | 0.9184 | 0.9201 | 0.9227 | 0.9257 | 0.9263 | 0.9282 |
| Aug-real | random | - | 0.9184 | 0.9184 | 0.9182 | 0.9183 | 0.9187 | 0.9179 |
| DiffGate | entropy | 0.933 | 0.8793 | 0.8850 | 0.8940 | 0.9084 | 0.9245 | 0.9381 |
| DiffGate | H1 | 0.662 | 0.8793 | 0.8859 | 0.8928 | 0.8986 | 0.9012 | 0.9024 |
| DiffGate | -r | 0.607 | 0.8793 | 0.8816 | 0.8755 | 0.8703 | 0.8692 | 0.8630 |
| DiffGate | random | - | 0.8793 | 0.8795 | 0.8794 | 0.8796 | 0.8798 | 0.8784 |
| SevGate | entropy | 0.876 | 0.8762 | 0.8801 | 0.8910 | 0.9059 | 0.9199 | 0.9343 |
| SevGate | H1 | 0.662 | 0.8762 | 0.8824 | 0.8890 | 0.8947 | 0.8971 | 0.8990 |
| SevGate | -r | 0.915 | 0.8762 | 0.8784 | 0.8825 | 0.8931 | 0.9015 | 0.9093 |
| SevGate | random | - | 0.8762 | 0.8762 | 0.8760 | 0.8766 | 0.8766 | 0.8755 |
| SevGate-real | entropy | 0.851 | 0.9149 | 0.9254 | 0.9329 | 0.9390 | 0.9443 | 0.9514 |
| SevGate-real | H1 | 0.662 | 0.9149 | 0.9166 | 0.9197 | 0.9225 | 0.9236 | 0.9258 |
| SevGate-real | -r | 0.999 | 0.9149 | 0.9166 | 0.9188 | 0.9227 | 0.9284 | 0.9360 |
| SevGate-real | random | - | 0.9149 | 0.9148 | 0.9146 | 0.9149 | 0.9150 | 0.9145 |

## Setting chapman (N = 1022)

| model | clean | seen | unseen | mixed | all corr. | NSTDB all | bw | emg | motion | dropout | powerline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Clean | 0.9958 | 0.9333 | 0.9335 | 0.8601 | 0.9212 | 0.8756 | 0.9601 | 0.9065 | 0.9592 | 0.9230 | 0.9183 |
| Aug | 0.9957 | 0.9840 | 0.9650 | 0.9617 | 0.9708 | 0.9083 | 0.9864 | 0.9816 | 0.9696 | 0.9468 | 0.9786 |
| DiffGate | 0.9942 | 0.9831 | 0.9609 | 0.9565 | 0.9676 | 0.9032 | 0.9834 | 0.9827 | 0.9628 | 0.9459 | 0.9740 |
| SevGate | 0.9948 | 0.9856 | 0.9659 | 0.9635 | 0.9721 | 0.9082 | 0.9862 | 0.9851 | 0.9660 | 0.9508 | 0.9808 |
| DiffGate-clean | 0.9940 | 0.8988 | 0.9228 | 0.8585 | 0.9041 | 0.8826 | 0.9561 | 0.8415 | 0.9556 | 0.9242 | 0.8886 |
| Aug-real | 0.9952 | 0.9784 | 0.9636 | 0.9598 | 0.9679 | 0.9668 | 0.9812 | 0.9756 | 0.9724 | 0.9468 | 0.9716 |
| SevGate-real | 0.9937 | 0.9797 | 0.9640 | 0.9607 | 0.9687 | 0.9691 | 0.9804 | 0.9791 | 0.9725 | 0.9486 | 0.9711 |

Per seed (mean +- s.d.): Clean mixed@-6 0.8521 +- 0.0168; Aug mixed@-6 0.9580 +- 0.0029; DiffGate mixed@-6 0.9496 +- 0.0014; SevGate mixed@-6 0.9580 +- 0.0029; DiffGate-clean mixed@-6 0.8424 +- 0.0278; Aug-real mixed@-6 0.9531 +- 0.0054; SevGate-real mixed@-6 0.9521 +- 0.0100

| comparison | endpoint | delta [95% CI] | p (one-sided) | TOST equiv. | d_z (seeds) |
|---|---|---|---|---|---|
| DiffGate - Aug | mixed@-6 | -0.0051 [-0.0079, -0.0026] | 1.000 | False | -5.38 (n=3) |
| DiffGate - Aug | unseen@-6 | -0.0041 [-0.0060, -0.0021] | 1.000 | False | -1.39 (n=3) |
| DiffGate - Aug | clean | -0.0015 [-0.0033, -0.0002] | 0.992 | True | -2.08 (n=3) |
| DiffGate - Aug | nstdb_all@-6 | -0.0051 [-0.0080, -0.0022] | 0.999 | False | -5.70 (n=3) |
| SevGate - Aug | mixed@-6 | +0.0018 [-0.0012, +0.0046] | 0.134 | True | -0.03 (n=3) |
| SevGate - Aug | unseen@-6 | +0.0009 [-0.0011, +0.0029] | 0.215 | True | -0.16 (n=3) |
| SevGate - Aug | clean | -0.0009 [-0.0027, +0.0004] | 0.898 | True | -0.52 (n=3) |
| SevGate - Aug | nstdb_all@-6 | -0.0001 [-0.0032, +0.0027] | 0.540 | True | -0.37 (n=3) |
| SevGate-real - Aug-real | nstdb_all@-6 | +0.0023 [+0.0001, +0.0046] | 0.020 | True | -0.03 (n=3) |

Share of the augmentation gain added by the gate: DiffGate mixed@-6 -0.05; DiffGate unseen@-6 -0.13; SevGate mixed@-6 +0.02; SevGate unseen@-6 +0.03

r diagnostics: DiffGate: clean 0.259, -6 dB 0.212, Spearman(SNR, r) +0.213; SevGate: clean 0.474, -6 dB 0.239, Spearman(SNR, r) +0.542; DiffGate-clean: clean 0.218, -6 dB 0.195, Spearman(SNR, r) +0.105; SevGate-real: clean 0.399, -6 dB 0.194, Spearman(SNR, r) +0.558

Clamp sweep (seed-averaged; r = 1 uses F_local only):

| model | r | clean | mixed@-6 | unseen@-6 | NSTDB all@-6 |
|---|---|---|---|---|---|
| DiffGate | learned | 0.9942 | 0.9565 | 0.9609 | 0.9032 |
| DiffGate | r=0 | 0.9926 | 0.9526 | 0.9603 | 0.9107 |
| DiffGate | r=0.25 | 0.9938 | 0.9530 | 0.9550 | 0.8967 |
| DiffGate | r=0.5 | 0.9922 | 0.9274 | 0.9296 | 0.8578 |
| DiffGate | r=0.75 | 0.9430 | 0.8207 | 0.8201 | 0.7235 |
| DiffGate | r=1 | 0.4471 | 0.4489 | 0.4404 | 0.4272 |
| DiffGate | learned - best const. | +0.0004 | +0.0035 | +0.0006 | -0.0075 |

DiffGate: a constant r within 0.002 of learned on every condition: False

| SevGate | learned | 0.9948 | 0.9635 | 0.9659 | 0.9082 |
| SevGate | r=0 | 0.9938 | 0.9621 | 0.9644 | 0.9080 |
| SevGate | r=0.25 | 0.9941 | 0.9600 | 0.9640 | 0.9054 |
| SevGate | r=0.5 | 0.9938 | 0.9508 | 0.9595 | 0.8958 |
| SevGate | r=0.75 | 0.9890 | 0.9137 | 0.9319 | 0.8711 |
| SevGate | r=1 | 0.7693 | 0.6753 | 0.6853 | 0.6799 |
| SevGate | learned - best const. | +0.0008 | +0.0014 | +0.0015 | +0.0002 |

SevGate: a constant r within 0.002 of learned on every condition: True


Selective prediction, 50/50 clean / synthetic mixed|-6|whole:

| model | score | detect AUROC | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|---|---|
| Aug | entropy | 0.909 | 0.9715 | 0.9781 | 0.9852 | 0.9908 | 0.9948 | 0.9989 |
| Aug | H1 | 0.500 | 0.9715 | 0.9705 | 0.9741 | 0.9701 | 0.9629 | 0.9580 |
| Aug | random | - | 0.9715 | 0.9716 | 0.9715 | 0.9712 | 0.9711 | 0.9708 |
| Aug-real | entropy | 0.935 | 0.9684 | 0.9748 | 0.9819 | 0.9875 | 0.9923 | 0.9969 |
| Aug-real | H1 | 0.500 | 0.9684 | 0.9684 | 0.9726 | 0.9681 | 0.9605 | 0.9543 |
| Aug-real | random | - | 0.9684 | 0.9686 | 0.9683 | 0.9677 | 0.9678 | 0.9675 |
| DiffGate | entropy | 0.905 | 0.9674 | 0.9767 | 0.9821 | 0.9868 | 0.9905 | 0.9950 |
| DiffGate | H1 | 0.500 | 0.9674 | 0.9664 | 0.9702 | 0.9666 | 0.9606 | 0.9551 |
| DiffGate | -r | 0.765 | 0.9674 | 0.9675 | 0.9687 | 0.9718 | 0.9692 | 0.9563 |
| DiffGate | random | - | 0.9674 | 0.9677 | 0.9676 | 0.9671 | 0.9669 | 0.9666 |
| SevGate | entropy | 0.879 | 0.9708 | 0.9779 | 0.9829 | 0.9891 | 0.9914 | 0.9982 |
| SevGate | H1 | 0.500 | 0.9708 | 0.9694 | 0.9718 | 0.9685 | 0.9629 | 0.9580 |
| SevGate | -r | 0.999 | 0.9708 | 0.9748 | 0.9776 | 0.9858 | 0.9905 | 0.9923 |
| SevGate | random | - | 0.9708 | 0.9708 | 0.9707 | 0.9709 | 0.9703 | 0.9694 |
| SevGate-real | entropy | 0.912 | 0.9611 | 0.9696 | 0.9766 | 0.9852 | 0.9887 | 0.9918 |
| SevGate-real | H1 | 0.500 | 0.9611 | 0.9607 | 0.9639 | 0.9596 | 0.9526 | 0.9476 |
| SevGate-real | -r | 1.000 | 0.9611 | 0.9675 | 0.9716 | 0.9804 | 0.9865 | 0.9895 |
| SevGate-real | random | - | 0.9611 | 0.9612 | 0.9610 | 0.9606 | 0.9604 | 0.9592 |

Selective prediction, 50/50 clean / nstdb_mixed|-6|whole:

| model | score | detect AUROC | 100% | 90% | 80% | 70% | 60% | 50% |
|---|---|---|---|---|---|---|---|---|
| Aug | entropy | 0.897 | 0.9008 | 0.9098 | 0.9254 | 0.9498 | 0.9709 | 0.9880 |
| Aug | H1 | 0.618 | 0.9008 | 0.9135 | 0.9225 | 0.9288 | 0.9303 | 0.9183 |
| Aug | random | - | 0.9008 | 0.9002 | 0.9004 | 0.9003 | 0.9006 | 0.8999 |
| Aug-real | entropy | 0.938 | 0.9731 | 0.9805 | 0.9855 | 0.9900 | 0.9939 | 0.9961 |
| Aug-real | H1 | 0.618 | 0.9731 | 0.9751 | 0.9760 | 0.9769 | 0.9753 | 0.9701 |
| Aug-real | random | - | 0.9731 | 0.9733 | 0.9729 | 0.9730 | 0.9728 | 0.9725 |
| DiffGate | entropy | 0.895 | 0.8874 | 0.8957 | 0.9136 | 0.9336 | 0.9598 | 0.9791 |
| DiffGate | H1 | 0.618 | 0.8874 | 0.8999 | 0.9092 | 0.9161 | 0.9194 | 0.9060 |
| DiffGate | -r | 0.730 | 0.8874 | 0.8777 | 0.8807 | 0.8823 | 0.8775 | 0.8909 |
| DiffGate | random | - | 0.8874 | 0.8870 | 0.8869 | 0.8870 | 0.8875 | 0.8856 |
| SevGate | entropy | 0.880 | 0.8873 | 0.8932 | 0.9118 | 0.9359 | 0.9579 | 0.9858 |
| SevGate | H1 | 0.618 | 0.8873 | 0.8999 | 0.9095 | 0.9169 | 0.9178 | 0.9047 |
| SevGate | -r | 0.988 | 0.8873 | 0.8967 | 0.9108 | 0.9356 | 0.9636 | 0.9922 |
| SevGate | random | - | 0.8873 | 0.8869 | 0.8867 | 0.8872 | 0.8869 | 0.8859 |
| SevGate-real | entropy | 0.892 | 0.9689 | 0.9757 | 0.9838 | 0.9897 | 0.9921 | 0.9953 |
| SevGate-real | H1 | 0.618 | 0.9689 | 0.9709 | 0.9708 | 0.9716 | 0.9700 | 0.9639 |
| SevGate-real | -r | 1.000 | 0.9689 | 0.9749 | 0.9795 | 0.9854 | 0.9899 | 0.9895 |
| SevGate-real | random | - | 0.9689 | 0.9691 | 0.9687 | 0.9686 | 0.9681 | 0.9679 |
