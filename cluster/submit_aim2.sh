#!/bin/bash
#SBATCH --job-name=aim2-pilot
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=05:00:00
#SBATCH --array=0-1
#SBATCH --output=cluster/logs/aim2_%A_%a.log
# task 0 = LFS arm (lambda=5), task 1 = domain-adaptation control (lambda=0)
set -o pipefail
cd "$HOME/multilingual-metrics"
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then LAM=5.0; TAG=aim2-lfs; else LAM=0.0; TAG=aim2-control; fi
python code/aim2_pilot.py --lam $LAM --out "runs/$TAG"
python code/cluster_grid.py --model "runs/$TAG" --n_sents 300 --batch_size 48 --dtype bf16 --device cuda
lm_eval --model hf --model_args "pretrained=runs/$TAG,dtype=bfloat16" --tasks belebele \
  --num_fewshot 0 --limit 300 --batch_size auto:4 \
  --output_path "results/belebele0/$TAG" --trust_remote_code
