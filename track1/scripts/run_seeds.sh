#!/bin/zsh
# Round 3 (fold 9 only): seeds 1-4 of the dataset-norm backbones + seed-0 variants of resnet1d_wang
# (SWA, EMA, label smoothing, crop 500). Rule for the final entries: results/crop/SELECTION2.md.
# Usage: scripts/run_seeds.sh wang|inception|xresnet50   (queues can run in parallel). Resumable.
set -e
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
O=results/crop
step() { echo "\n===== $(date '+%F %T') $* =====" }
run() { local tag=$1; shift; [[ -f $O/$tag.json ]] && echo "skip $tag" || { step $tag; python train.py --norm dataset --out $O --tag $tag "$@" }; }
seeds() { for s in 1 2 3 4; do run ${1}_crop_dsnorm_s$s --model $1 --crop 250 --seed $s; done }

case $1 in
  wang)
    seeds resnet1d_wang
    W=(--model resnet1d_wang --seed 0)
    run resnet1d_wang_crop_dsnorm_swa  $W --crop 250 --avg swa
    run resnet1d_wang_crop_dsnorm_ema  $W --crop 250 --avg ema
    run resnet1d_wang_crop_dsnorm_ls05 $W --crop 250 --label-smooth 0.05
    run resnet1d_wang_crop500_dsnorm   $W --crop 500 --stride 250 ;;
  inception) seeds inception1d ;;
  xresnet50) seeds xresnet1d50 ;;
  *) echo "usage: $0 wang|inception|xresnet50"; exit 1 ;;
esac
echo "QUEUE_DONE $1"
