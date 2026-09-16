#!/bin/bash
#SBATCH --job-name=mlm-bele0
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=cluster/logs/bele0_%A_%a.log

# Belebele (122 languages) x our 10-model grid via lm-evaluation-harness.
# 5-shot, log-likelihood MCQ scoring (matches MEXA's protocol), 300 Qs/language.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/belebele

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODELS=(
  "Qwen/Qwen3-0.6B-Base"
  "Qwen/Qwen3-1.7B-Base"
  "Qwen/Qwen3-4B-Base"
  "Qwen/Qwen3-8B-Base"
  "allenai/OLMo-2-0425-1B"
  "allenai/OLMo-2-1124-7B"
  "mistralai/Mistral-7B-v0.3"
  "bigscience/bloom-1b7"
  "bigscience/bloom-7b1"
  "utter-project/EuroLLM-1.7B"
)
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
