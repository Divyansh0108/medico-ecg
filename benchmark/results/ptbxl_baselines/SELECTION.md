# Final selection (fixed on fold 9 BEFORE any fold-10 evaluation), 2026-09-30

Recipe: 2.5 s random crops, sliding-window mean (250/125), band-pass on, mixup 0.4, Adam 5e-4 cosine,
dataset-level standardization (train-fitted global mean/std; +0.016-0.019 fold-9 for every backbone).

- FINAL_ensemble: mean probabilities of all 5 dataset-norm runs (M1, resnet1d_wang, xresnet1d50,
  xresnet1d101, inception1d). Pre-specified rule, NOT the best fold-9 subset: top subsets differ by <0.0005
  (noise), and searching over them would bias fold 10.
- FINAL_single: best fold-9 single model, resnet1d_wang_crop_dsnorm (0.9365), for comparison with the
  published single-model rows.
Each evaluated once on fold 10 via final_eval.py.
