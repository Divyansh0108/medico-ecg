#!/bin/zsh
# Revision ablations (RULES3.md 1). Usage: rules3_ablation_train.sh QUEUE (a|b). Resumable (skips finished tags).
set -e
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
O=results/robustness/runs
step() { echo "\n===== $(date '+%F %T') $* =====" }
run() { local tag=$1; shift; [[ -f $O/$tag.json ]] && echo "skip $tag" || { step $tag; python -m training.train --norm dataset --crop 250 --out $O --tag $tag "$@" }; }
for s in 0 1 2; do
  if [[ $1 == a ]]; then
    run Fm002_s$s    --seed $s --model t2_E --aux --sev-margin 0.02
    run Fm010_s$s    --seed $s --model t2_E --aux --sev-margin 0.10
    run Fws03_s$s    --seed $s --model t2_E --aux --w-sev 0.3
    run Fwc0_s$s     --seed $s --model t2_E --aux --w-cons 0
    run B0augall_s$s --seed $s --model resnet1d_wang --aug --aug-families all
  else
    run Fh128_s$s    --seed $s --model t2_E_h128 --aux
    run T_s$s        --seed $s --model t2_T --aug
    run TF_s$s       --seed $s --model t2_T --aux
    run Fall_s$s     --seed $s --model t2_E --aux --aug-families all
  fi
done
echo "QUEUE_DONE $1"
