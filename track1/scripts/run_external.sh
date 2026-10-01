#!/bin/zsh
# After the single-lead queues: CinC2017 and BUT QDB inference + reports. After the real-noise queues:
# NSTDB fold-9 evaluation of B0-aug-real / F-real and the NSTDB report. No training, no fold 10.
cd "$(dirname "$0")/.."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate medico
export PYTHONUNBUFFERED=1
until [ "$(cat logs/sl_train_s*.log | grep -c QUEUE_DONE)" -ge 3 ]; do sleep 30; done
python cinc2017.py && python cinc2017_report.py > /dev/null && echo CINC_DONE
python butqdb.py infer && python butqdb_report.py > /dev/null && echo BUTQDB_DONE
until [ "$(cat logs/real_train_s*.log | grep -c QUEUE_DONE)" -ge 3 ]; do sleep 30; done
python track2_eval.py --name nstdb --nstdb-snrs 0 -6 --tags B0augreal_s0 B0augreal_s1 B0augreal_s2 Freal_s0 Freal_s1 Freal_s2 \
  && python nstdb_report.py > /dev/null && echo REAL_DONE
