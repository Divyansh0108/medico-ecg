#!/bin/zsh
# Overnight pipeline for PTB-XL Track 1. Resumable: skips runs whose results JSON exists.
set -e
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
step() { echo "\n===== $(date '+%F %T') $* =====" }
run() { [[ -f results/$1_s$2.json ]] && echo "skip $1_s$2 (exists)" || python train.py --model $1 --seed $2 --eval-test }

step "STAGE 0: data check";           python check_data.py 2>&1 | grep -v "load "
step "STAGE 1: M1 seed 0";            run M1 0
step "STAGE 2: M2 seed 0";            run M2 0
step "COMPARISON";                    python compare.py | tee results/comparison_s0.txt
step "STAGE 3: calibration";          python calibrate.py
chosen=$(python decide.py)
if [[ -z "$chosen" ]]; then
  step "GATE: best seed-0 test mAUROC < 0.92 -> stop (no seeds 1-4)"; echo PIPELINE_DONE; exit 0
fi
step "GATE passed -> seeds 1-4 for: $chosen"
for m in ${=chosen}; do for s in 1 2 3 4; do step "STAGE 4: $m seed $s"; run $m $s; done; done
step "STAGE 4: seed summary + ensemble"; python ensemble.py ${=chosen} | tee results/seeds_ensemble.txt
echo PIPELINE_DONE
