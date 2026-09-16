#!/bin/bash
#SBATCH --job-name=mlm-llama
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=cluster/logs/llama_%A_%a.log

# A3 third family: Llama-3.1-8B (predictions P-A3.5/6 frozen before this
# submission; gated access via user token in ~/.cache/huggingface/token).
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/grid results/belebele0

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

M="meta-llama/Llama-3.1-8B"
nvidia-smi --query-gpu=name --format=csv,noheader || true
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  python code/cluster_grid.py --model "$M" --n_sents 300 --batch_size 16 --dtype bf16
else
  lm_eval --model hf \
    --model_args "pretrained=$M,dtype=bfloat16" \
    --tasks belebele --num_fewshot 0 --limit 300 --batch_size auto:4 \
    --output_path "results/belebele0/Llama-3.1-8B" --trust_remote_code
fi
