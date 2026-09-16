#!/bin/bash
#SBATCH --job-name=mlm-grid
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-10
#SBATCH --output=cluster/logs/grid_%A_%a.log

# LFM+AaR metric grid: one model per array task (11 models, 1 GPU each).
set -eo pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/grid

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments

# Overlay venv: newer transformers WITHOUT touching the shared conda env
VENV="$HOME/venvs/mlm"
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV" --system-site-packages
    "$VENV/bin/pip" install -q -U "transformers>=4.51" accelerate sentencepiece protobuf
fi
source "$VENV/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

python code/cluster_grid.py --model_idx "$SLURM_ARRAY_TASK_ID" --n_sents 300 --batch_size 32
