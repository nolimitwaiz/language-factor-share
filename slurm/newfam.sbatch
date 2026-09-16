#!/bin/bash
#SBATCH --job-name=mlm-newfam
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=cluster/logs/newfam_%A_%a.log

# A3 families (predictions frozen 2026-07-19 BEFORE this submission):
# granite-3.1-8b-base + Yi-1.5-9B. Tasks 0-1 = grid fingerprints,
# tasks 2-3 = 0-shot Belebele (exam-capability scoring, frozen 0.30 rule).
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/grid results/belebele0

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODELS=("ibm-granite/granite-3.1-8b-base" "01-ai/Yi-1.5-9B")
M=${MODELS[$((SLURM_ARRAY_TASK_ID % 2))]}
TAG=$(basename "$M")
nvidia-smi --query-gpu=name --format=csv,noheader || true

if [ "$SLURM_ARRAY_TASK_ID" -lt 2 ]; then
  echo "[grid] $M"
  python code/cluster_grid.py --model "$M" --n_sents 300 --batch_size 16 --dtype bf16
else
  echo "[belebele] $M"
  lm_eval --model hf \
    --model_args "pretrained=$M,dtype=bfloat16" \
    --tasks belebele --num_fewshot 0 --limit 300 --batch_size auto:4 \
    --output_path "results/belebele0/$TAG" --trust_remote_code
fi
