#!/bin/zsh
# Optional "trained on real noise" runs (RULES2.md 1): B0-aug-real and F-real for one seed, after that
# seed's single-lead queue finishes. Usage: SEED. Resumable.
set -e
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
O=results/track2/runs
s=$1
until grep -q "QUEUE_DONE $s" logs/sl_train_s$s.log; do sleep 30; done
step() { echo "\n===== $(date '+%F %T') $* =====" }
run() { local tag=$1; shift; [[ -f $O/$tag.json ]] && echo "skip $tag" || { step $tag; python train.py --norm dataset --crop 250 --seed $s --out $O --tag $tag "$@" }; }
run B0augreal_s$s --model resnet1d_wang --aug-real
run Freal_s$s     --model t2_E --aux --aug-real
echo "QUEUE_DONE $s"
