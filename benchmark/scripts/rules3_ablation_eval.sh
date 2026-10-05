#!/bin/zsh
# Waits for both revision queues, then evaluates the 27 ablation models on the fold-9 synthetic grid and
# the NSTDB eval grid (RULES3.md 2), links the reference Aug/SevGate predictions and writes the report.
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
until [ "$(cat logs/revision_[ab].log | grep -c QUEUE_DONE)" -ge 2 ]; do
  pgrep -f rules3_ablation_train.sh >/dev/null || { echo "training queues died"; exit 1; }; sleep 60; done
T=()
for p in Fm002 Fm010 Fws03 Fwc0 Fh128 T TF B0augall Fall; do for s in 0 1 2; do T+=(${p}_s$s); done; done
python -m evaluation.robustness_eval --name ablation --snrs 15 6 0 -6 --tags $T && \
python -m evaluation.robustness_eval --name ablation_nstdb --nstdb-snrs 0 -6 --tags $T || exit 1
for s in 0 1 2; do for p in B0aug F; do
  ln -sf ../main/${p}_s$s.npz results/robustness/grid/ablation/${p}_s$s.npz
  ln -sf ../nstdb/${p}_s$s.npz results/robustness/grid/ablation_nstdb/${p}_s$s.npz
done; done
python -m reporting.rules3_ablation_report > /dev/null && echo REVISION_EVAL_DONE
