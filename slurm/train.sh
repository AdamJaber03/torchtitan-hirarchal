#!/bin/bash
# One SLURM job of a named run from my_recipes.py on GOLEM L40S nodes (8 GPUs
# each). Submit from a login node; jobs submitted from inside another job
# (e.g. the VS Code session) fail to launch:
#   ssh login02 'cd ~/torchtitan-hirarchal && sbatch -N 2 slurm/train.sh qwen3_0_6b_fineweb_edu_run'
# The config is an argument, not --export=ALL,CONFIG=...: on GOLEM any
# --export=ALL,VAR=value fails with "user env retrieval failed" (job held).
# Multi-node runs use HSDP: shard within a node, replicate across nodes (see
# _hsdp_for_launch in my_recipes.py). Relaunching the same CONFIG resumes from
# its latest checkpoint and W&B run; keep the node count fixed across a run,
# since the dataloader refuses to resume with a different DP size.
# Usage: slurm/train.sh CONFIG [extra torchtitan.train args...]
# Env (inherited from the submitting shell): MODULE=my_recipes, NGPU=8,
#      LOG_RANK=0, TORCHTITAN_OUTPUT_ROOT (see my_recipes.py).
#SBATCH --job-name=torchtitan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_l40s:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

CONFIG=${1:?usage: sbatch [-N NODES] slurm/train.sh CONFIG [extra args...]}
shift
MODULE=${MODULE:-my_recipes}
NGPU=${NGPU:-8}
REPO=${REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=$((29500 + SLURM_JOB_ID % 1000))

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONFAULTHANDLER=1
# Log to W&B online only when logged in (wandb login writes ~/.netrc);
# otherwise keep the run offline and sync it later with `wandb sync`.
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -qs api.wandb.ai ~/.netrc; then
    echo "W&B: no credentials found, logging offline" >&2
    export WANDB_MODE=${WANDB_MODE:-offline}
fi

echo "job $SLURM_JOB_ID: $SLURM_NNODES node(s) x $NGPU GPUs [$SLURM_JOB_NODELIST], config $MODULE.$CONFIG"
srun --kill-on-bad-exit=1 --cpu-bind=none \
    .venv/bin/torchrun --nnodes "$SLURM_NNODES" --nproc_per_node "$NGPU" \
    --rdzv_id "$SLURM_JOB_ID" --rdzv_backend c10d --rdzv_endpoint "$head_node:$port" \
    --local-ranks-filter "${LOG_RANK:-0}" --role rank --tee 3 \
    -m torchtitan.train --module "$MODULE" --config "$CONFIG" "$@"
