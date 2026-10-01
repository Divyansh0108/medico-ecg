#!/bin/zsh
# Single-lead (PTB-XL lead I) variants for CinC2017 / BUT QDB (RULES2.md section 3A). Usage: SEED. Resumable.
set -e
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
O=results/track2/runs_sl
s=$1
step() { echo "\n===== $(date '+%F %T') $* =====" }
run() { local tag=$1; shift; [[ -f $O/$tag.json ]] && echo "skip $tag" || { step $tag; python train.py --leads 0 --norm dataset --crop 250 --seed $s --out $O --tag $tag "$@" }; }
run SL_B0clean_s$s --model resnet1d_wang
run SL_B0aug_s$s   --model resnet1d_wang --aug
run SL_D_s$s       --model t2_D --aug
run SL_E_s$s       --model t2_E --aug
run SL_F_s$s       --model t2_E --aux
run SL_Eclean_s$s  --model t2_E
echo "QUEUE_DONE $s"
