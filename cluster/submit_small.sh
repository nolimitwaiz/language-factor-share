#!/bin/bash
#SBATCH --job-name=mlm-small
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --array=0,1,2,4,7,9
#SBATCH --output=cluster/logs/small_%A_%a.log

# Small/mid models (<=4B) on the general GPU partition (RTX-class, fp16).
# Big models (7-9B) stay on the queued 3xA100 job.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/grid

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python code/cluster_grid.py --model_idx "$SLURM_ARRAY_TASK_ID" \
    --n_sents 300 --batch_size 16 --dtype fp16
