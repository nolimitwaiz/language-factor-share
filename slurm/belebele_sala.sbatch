#!/bin/bash
#SBATCH --job-name=mlm-bele-sala
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=cluster/logs/bele_sala_%A_%a.log

# Belebele 0-shot for the Salamandra family: adds a 5th exam-capable
# candidate family to the deflation/validity analysis (the "add more capable
# families" hardening item). Protocol identical to the grid's belebele0 runs.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/belebele0

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODELS=("BSC-LT/salamandra-2b" "BSC-LT/salamandra-7b")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
TAG=$(basename "$MODEL")
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "[belebele] $MODEL"

lm_eval --model hf \
  --model_args "pretrained=$MODEL,dtype=bfloat16" \
  --tasks belebele \
  --num_fewshot 0 \
  --limit 300 \
  --batch_size auto:4 \
  --output_path "results/belebele0/$TAG" \
  --trust_remote_code
