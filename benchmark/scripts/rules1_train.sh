#!/bin/zsh
# Track 2 training (fold 9 early stopping on clean data only). Usage: scripts/rules1_train.sh SEED
# Runs B0-aug, C, D, E, F (all aug regime) and E-clean for one seed. Resumable.
set -e
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
O=results/robustness/runs
s=$1
step() { echo "\n===== $(date '+%F %T') $* =====" }
run() { local tag=$1; shift; [[ -f $O/$tag.json ]] && echo "skip $tag" || { step $tag; python -m training.train --norm dataset --crop 250 --seed $s --out $O --tag $tag "$@" }; }
run B0aug_s$s  --model resnet1d_wang --aug
run C_s$s      --model t2_C --aug
run D_s$s      --model t2_D --aug
run E_s$s      --model t2_E --aug
run F_s$s      --model t2_E --aux
run Eclean_s$s --model t2_E
echo "QUEUE_DONE $s"
