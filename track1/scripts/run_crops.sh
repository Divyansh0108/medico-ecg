#!/bin/zsh
# Crops + Strodthoff backbones + M1 ablations (seed 0). Fold 9 only, except M1_crop fold 10 (asked explicitly).
# Resumable: skips runs whose results/crop/{tag}.json exists.
set -e
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
O=results/crop
step() { echo "\n===== $(date '+%F %T') $* =====" }
run() { local tag=$1; shift; [[ -f $O/$tag.json ]] && echo "skip $tag" || { step $tag; python train.py --crop 250 --seed 0 --out $O --tag $tag "$@" }; }

run M1_crop                --model M1
[[ -f $O/M1_crop_fold10.json ]] || { step "M1_crop fold 10"; python final_eval.py --name M1_crop_fold10 --tags M1_crop; }
for m in resnet1d_wang xresnet1d50 xresnet1d101 inception1d; do run ${m}_crop --model $m; done
run M1_crop_nomixup        --model M1 --mixup 0
run M1_crop_nobandpass     --model M1 --no-bandpass
run M1_crop_datasetnorm    --model M1 --norm dataset
run M1_crop_onecycle       --model M1 --sched onecycle --lr 1e-2
step REPORT; python crop_report.py
echo CROPS_DONE
# Stage 2 (added after fold-9 ablations: dataset-level norm +0.018 on M1, mostly HYP): backbones with dataset norm.
for m in resnet1d_wang xresnet1d50 xresnet1d101 inception1d; do run ${m}_crop_dsnorm --model $m --norm dataset; done
step REPORT2; python crop_report.py
echo STAGE2_DONE
