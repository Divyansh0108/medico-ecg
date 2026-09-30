#!/bin/zsh
# Waits for the three training queues, then evaluates all 21 models on the fold-9 grid (4 SNR levels,
# RULES.md section E) and writes the report + verdict. Fold 10 is not touched.
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
until [ "$(cat logs/t2_train_s*.log | grep -c QUEUE_DONE)" -ge 3 ]; do
  pgrep -f run_track2.sh >/dev/null || { echo "training queues died"; exit 1; }; sleep 30; done
T=(resnet1d_wang_crop_dsnorm resnet1d_wang_crop_dsnorm_s1 resnet1d_wang_crop_dsnorm_s2)
for p in B0aug C D E F Eclean; do for s in 0 1 2; do T+=(${p}_s$s); done; done
python track2_eval.py --name main --snrs 15 6 0 -6 --tags $T && python track2_report.py > /dev/null && echo EVAL_REPORT_DONE
