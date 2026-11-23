#!/bin/bash
# Queue N chained jobs of one run: slurm/queue.sh CONFIG LOG_DIR [N=1] [NODES=1]
# Each job starts after the previous one ends (also on timeout or failure) and
# resumes the run. The gpu partition has no time limit, so N=1 is usually
# enough; chain more to survive the 3-day --time in slurm/train.sh.
# Run on a login node (ssh login02), not inside a job.
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "run this on a login node: ssh login02 'cd $PWD && slurm/queue.sh $*'" >&2
    exit 1
fi
CONFIG=${1:?CONFIG} LOG_DIR=${2:?LOG_DIR} N=${3:-1} NODES=${4:-1}
cd "$(dirname "$0")/.."
mkdir -p "$LOG_DIR"
dep=""
for i in $(seq 1 "$N"); do
    id=$(sbatch --parsable $dep -N "$NODES" --job-name="$CONFIG" \
        -o "$LOG_DIR/%j.out" slurm/train.sh "$CONFIG")
    echo "queued job $i/$N: $id ($NODES node(s))"
    dep="--dependency=afterany:$id"
done
