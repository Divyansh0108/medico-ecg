#!/bin/zsh
# After the single-lead queues: CinC2017 and BUT QDB inference + reports. After the real-noise queues:
# NSTDB fold-9 evaluation of B0-aug-real / F-real and the NSTDB report. No training, no fold 10.
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
until [ "$(cat logs/sl_train_s*.log | grep -c QUEUE_DONE)" -ge 3 ]; do sleep 30; done
python -m external.cinc2017 && python -m reporting.cinc2017_report > /dev/null && echo CINC_DONE
python -m external.butqdb infer && python -m reporting.butqdb_report > /dev/null && echo BUTQDB_DONE
until [ "$(cat logs/real_train_s*.log | grep -c QUEUE_DONE)" -ge 3 ]; do sleep 30; done
python -m evaluation.robustness_eval --name nstdb --nstdb-snrs 0 -6 --tags B0augreal_s0 B0augreal_s1 B0augreal_s2 Freal_s0 Freal_s1 Freal_s2 \
  && python -m reporting.nstdb_report > /dev/null && echo REAL_DONE
