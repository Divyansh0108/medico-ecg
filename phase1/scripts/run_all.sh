#!/usr/bin/env bash
# Full Phase 1 sweep, sequential: config `variants` x {clean, aug} (paired F, G: aug only) x seeds
# (Phase 1: A, C, E x 2 regimes x 5 seeds = 30 runs). Complete runs (best.pt + meta.json) are
# skipped, so the sweep is resumable.
#   scripts/run_all.sh            # full sweep
#   scripts/run_all.sh --smoke    # same sweep on synthetic data with the config's smoke overrides
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/phase1.yaml}"
EXTRA=("$@")
PY=(conda run --no-capture-output -n medico python)

SEEDS=$("${PY[@]}" -c "import sys, train; print(' '.join(map(str, train.load_config(sys.argv[1], '--smoke' in sys.argv[2:])['train']['seeds'])))" "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"})

JOBS=()
while read -r v r; do JOBS+=("$v $r"); done < <("${PY[@]}" -c "import sys, train; [print(v, r) for v, r in train.sweep_jobs(train.load_config(sys.argv[1], '--smoke' in sys.argv[2:]))]" "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"})

N=$(( ${#JOBS[@]} * $(wc -w <<<"$SEEDS") ))
echo "sweep: ${#JOBS[@]} variant/regime pairs x seeds [$SEEDS] = $N runs"
i=0
for s in $SEEDS; do
  for job in "${JOBS[@]}"; do
    read -r v r <<<"$job"
    i=$((i + 1))
    echo "=== [$i/$N] variant=$v regime=$r seed=$s"
    "${PY[@]}" train.py --config "$CONFIG" --variant "$v" --regime "$r" --seed "$s" ${EXTRA[@]+"${EXTRA[@]}"}
  done
done
echo "sweep finished: $N runs"
