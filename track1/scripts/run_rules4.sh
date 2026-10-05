#!/bin/zsh
# RULES4.md section 1 runs. Usage: run_rules4.sh QUEUE (a: xresnet1d50, b: inception1d, c: wang seeds 3-4 then
# Chapman). Resumable (skips finished tags). Queue c waits for the Chapman download (DL_DONE in $DL_LOG).
set -e
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
O=results/track2/runs
step() { echo "\n===== $(date '+%F %T') $* =====" }
run() { local tag=$1; shift; [[ -f $O/$tag.json ]] && echo "skip $tag" || { step $tag; python train.py --norm dataset --crop 250 --out $O --tag $tag "$@" }; }
arch() {   # prefix, plain trunk, gated trunk
  for s in 0 1 2; do
    run $1B0aug_s$s     --seed $s --model $2 --aug
    run $1E_s$s         --seed $s --model $3 --aug
    run $1F_s$s         --seed $s --model $3 --aux
    run $1Eclean_s$s    --seed $s --model $3
    run $1B0augreal_s$s --seed $s --model $2 --aug-real
    run $1Freal_s$s     --seed $s --model $3 --aux --aug-real
  done
}
case $1 in
  a) arch X xresnet1d50 t2x_E ;;
  b) arch I inception1d t2i_E ;;
  c)
    for s in 3 4; do
      run B0aug_s$s --seed $s --model resnet1d_wang --aug
      run D_s$s     --seed $s --model t2_D --aug
      run E_s$s     --seed $s --model t2_E --aug
      run F_s$s     --seed $s --model t2_E --aux
    done
    until grep -q DL_DONE $DL_LOG; do sleep 60; done
    for s in 0 1 2; do
      run CHB0clean_s$s  --seed $s --dataset chapman --model resnet1d_wang
      run CHB0aug_s$s    --seed $s --dataset chapman --model resnet1d_wang --aug
      run CHE_s$s        --seed $s --dataset chapman --model t2_E --aug
      run CHF_s$s        --seed $s --dataset chapman --model t2_E --aux
      run CHEclean_s$s   --seed $s --dataset chapman --model t2_E
      run CHB0augreal_s$s --seed $s --dataset chapman --model resnet1d_wang --aug-real
      run CHFreal_s$s    --seed $s --dataset chapman --model t2_E --aux --aug-real
    done ;;
esac
echo "QUEUE_DONE $1"
