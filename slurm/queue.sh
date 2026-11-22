#!/bin/bash
# Queue N chained 12 h jobs of one run: slurm/queue.sh CONFIG RUN_DIR N
# Each job starts after the previous one ends (also on timeout) and resumes it.
set -e
CONFIG=$1 RUN_DIR=$2 N=${3:-4}
mkdir -p "$RUN_DIR/slurm"
dep=""
for i in $(seq 1 "$N"); do
  id=$(sbatch --parsable $dep --job-name="$(basename "$RUN_DIR")" \
    --export=ALL,CONFIG="$CONFIG" -o "$RUN_DIR/slurm/%j.out" slurm/train.sh)
  echo "queued job $i/$N: $id"
  dep="--dependency=afterany:$id"
done
