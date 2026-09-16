#!/bin/bash
#SBATCH --job-name=mlm-r2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=cluster/logs/r2_%A_%a.log

# §3.1.2 lens + §3.1.3 transfer: tasks 0-3 = lens, 4-7 = transfer.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/lens312 results/transfer313

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python -c "import scipy" 2>/dev/null || pip install -q scipy

MODELS=("Qwen/Qwen3-0.6B-Base" "Qwen/Qwen3-1.7B-Base" "Qwen/Qwen3-4B-Base" "allenai/OLMo-2-0425-1B")
IDX=$((SLURM_ARRAY_TASK_ID % 4))
nvidia-smi --query-gpu=name --format=csv,noheader || true

if [ "$SLURM_ARRAY_TASK_ID" -lt 4 ]; then
    python code/lens_312.py --model "${MODELS[$IDX]}" --device cuda --pool mean
else
    python code/transfer_313b.py --model "${MODELS[$IDX]}" --device cuda
fi
