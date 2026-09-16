#!/bin/bash
#SBATCH --job-name=mlm-big
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --array=3,5,6,8,10
#SBATCH --output=cluster/logs/big_%A_%a.log

# Big models (7-9B), ONE GPU + SHORT walltime each -> backfill-friendly.
# idx: 3=Qwen3-8B 5=OLMo-2-7B 6=Mistral-7B 8=bloom-7b1 10=EuroLLM-9B
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
    --n_sents 300 --batch_size 48 --dtype bf16
