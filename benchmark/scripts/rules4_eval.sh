#!/bin/zsh
# RULES4.md sections 2 and 5: waits for the three rules4_train.sh queues, evaluates every setting on the synthetic
# grid and the NSTDB eval grid, runs the gate clamp sweep, then writes the report (reporting/rules4_report.py).
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
until [ "$(cat logs/rules4_[abc].log | grep -c QUEUE_DONE)" -ge 3 ]; do
  pgrep -f rules4_train.sh >/dev/null || { echo "training queues died"; exit 1; }; sleep 60; done
G=results/robustness/grid
ev() { local name=$1; shift; python -m evaluation.robustness_eval --name $name "$@" || exit 1 }
grids() {   # name, tags...
  local n=$1; shift
  ev r4_$n --snrs 15 6 0 -6 --tags "$@"
  ev r4_${n}_nstdb --nstdb-snrs 0 -6 --tags "$@"
}
tags() { local T=(); for p in "$@"; do for s in 0 1 2; do T+=(${p}_s$s); done; done; echo $T }
# S1 wang: new seeds 3-4 evaluated, seeds 0-2 linked from grid/main and grid/nstdb
W=(resnet1d_wang_crop_dsnorm_s3 resnet1d_wang_crop_dsnorm_s4)
for p in B0aug D E F; do W+=(${p}_s3 ${p}_s4); done
grids wang $W
for f in $G/main/*.npz; do ln -sf ../main/${f:t} $G/r4_wang/${f:t}; done
for f in $G/nstdb/*.npz; do ln -sf ../nstdb/${f:t} $G/r4_wang_nstdb/${f:t}; done
grids xresnet xresnet1d50_crop_dsnorm xresnet1d50_crop_dsnorm_s1 xresnet1d50_crop_dsnorm_s2 ${=$(tags XB0aug XE XF XEclean XB0augreal XFreal)}
grids inception inception1d_crop_dsnorm inception1d_crop_dsnorm_s1 inception1d_crop_dsnorm_s2 ${=$(tags IB0aug IE IF IEclean IB0augreal IFreal)}
grids chapman ${=$(tags CHB0clean CHB0aug CHE CHF CHEclean CHB0augreal CHFreal)}
# clamp sweep (RULES4.md 5)
for r in 0 0.25 0.5 0.75 1; do
  ev r4clamp_wang_r$r --snrs -6 --nstdb-snrs -6 --force-r $r --tags E_s{0..4} F_s{0..4}
  ev r4clamp_xresnet_r$r --snrs -6 --nstdb-snrs -6 --force-r $r --tags ${=$(tags XE XF)}
  ev r4clamp_inception_r$r --snrs -6 --nstdb-snrs -6 --force-r $r --tags ${=$(tags IE IF)}
  ev r4clamp_chapman_r$r --snrs -6 --nstdb-snrs -6 --force-r $r --tags ${=$(tags CHE CHF)}
done
python -m reporting.rules4_report > /dev/null && echo RULES4_EVAL_DONE
