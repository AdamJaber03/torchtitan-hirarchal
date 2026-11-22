#!/bin/bash
# One SLURM job of a named run from my_recipes.py. Relaunching the same CONFIG
# resumes from its latest checkpoint and W&B run, so queue a chain with
# slurm/queue.sh. Keep the GPU count fixed across a run.
#SBATCH --partition=long.q
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=48G
#SBATCH -c 2
#SBATCH --time=12:00:00
set -e
cd /home/adamga/torchtitan-hirarchal/torchtitan-hirarchal
source .venv/bin/activate
hostname
NGPU=2 MODULE=my_recipes CONFIG=${CONFIG:?set CONFIG} bash run_train.sh
