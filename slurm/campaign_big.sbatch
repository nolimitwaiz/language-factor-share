#!/bin/bash
#SBATCH --job-name=mlm-camp-b
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=3:00:00
#SBATCH --array=0-5
#SBATCH --output=cluster/logs/camp_b_%A_%a.log

# Campaign re-embed: 7-9B models on A100, bf16 (native for all; short
# walltime -> backfill).
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/dumps

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODELS=("Qwen/Qwen3-8B-Base" "allenai/OLMo-2-1124-7B" "mistralai/Mistral-7B-v0.3" \
        "bigscience/bloom-7b1" "tiiuae/Falcon3-7B-Base" "BSC-LT/salamandra-7b")
M=${MODELS[$SLURM_ARRAY_TASK_ID]}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python src/campaign/dump_embeddings.py --model "$M" --n_sents 1500 \
    --dtype bf16 --batch_size 16 --device cuda --save_margins
