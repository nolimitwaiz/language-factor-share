#!/bin/bash
#SBATCH --job-name=wf-big
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=cluster/logs/wf_%A_%a.log
set -o pipefail
cd "$HOME/multilingual-metrics"
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name --format=csv,noheader || true
M="tiiuae/Falcon3-7B-Base"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  python code/cluster_grid.py --model "$M" --n_sents 300 --batch_size 16 --dtype bf16 --device cuda
else
  lm_eval --model hf --model_args "pretrained=$M,dtype=bfloat16" --tasks belebele \
    --num_fewshot 0 --limit 300 --batch_size auto:4 \
    --output_path "results/belebele0/Falcon3-7B-Base" --trust_remote_code
fi
