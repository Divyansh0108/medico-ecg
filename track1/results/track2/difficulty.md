## Difficulty check (fold 9, B0-clean = resnet1d_wang_crop_dsnorm seeds 0-2, no retraining)

Clean macro-AUROC per seed: 0.9365 0.9356 0.9349 (mean 0.9356).
Cells: mean over the 3 seeds of macro-AUROC (drop vs clean).

| family | mode | +15 dB | +6 dB | +0 dB | -6 dB |
|---|---|---|---|---|---|
| baseline_wander | whole | 0.9354 (+0.000) | 0.9285 (+0.007) | 0.9075 (+0.028) | 0.8371 (+0.099) |
| baseline_wander | burst | 0.9355 (+0.000) | 0.9349 (+0.001) | 0.9323 (+0.003) | 0.9289 (+0.007) |
| emg | whole | 0.9357 (-0.000) | 0.9306 (+0.005) | 0.9098 (+0.026) | 0.8359 (+0.100) |
| emg | burst | 0.9358 (-0.000) | 0.9352 (+0.000) | 0.9329 (+0.003) | 0.9301 (+0.006) |
| motion_burst | whole | 0.9352 (+0.000) | 0.9282 (+0.007) | 0.9126 (+0.023) | 0.8640 (+0.072) |
| motion_burst | burst | 0.9355 (+0.000) | 0.9321 (+0.004) | 0.9315 (+0.004) | 0.9285 (+0.007) |
| dropout | whole | 0.9340 (+0.002) | 0.9258 (+0.010) | 0.8949 (+0.041) | 0.8119 (+0.124) |
| dropout | burst | 0.9355 (+0.000) | 0.9342 (+0.001) | 0.9323 (+0.003) | 0.9308 (+0.005) |
| powerline | whole | 0.9358 (-0.000) | 0.9331 (+0.003) | 0.9159 (+0.020) | 0.8446 (+0.091) |
| powerline | burst | 0.9357 (-0.000) | 0.9355 (+0.000) | 0.9337 (+0.002) | 0.9301 (+0.006) |
| mixed | whole | 0.9352 (+0.000) | 0.9256 (+0.010) | 0.8949 (+0.041) | 0.7860 (+0.150) |
| mixed | burst | 0.9355 (+0.000) | 0.9345 (+0.001) | 0.9308 (+0.005) | 0.9281 (+0.008) |

Severity rule: mean drop on mixed (3 seeds x 2 modes): +15 dB 0.0003, +6 dB 0.0056, +0 dB 0.0228, -6 dB 0.0786; bar 0.03.
**Mildest SNR meeting the bar: -6 dB. Chosen severity set: -6 dB.**
