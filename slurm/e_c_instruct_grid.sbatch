#!/bin/bash
#SBATCH --job-name=aim1-ec-instruct
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --array=0-6%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --output=cluster/logs/ec_instruct_%A_%a.out
#SBATCH --error=cluster/logs/ec_instruct_%A_%a.err
set -eo pipefail
cd "$HOME/multilingual-metrics"; mkdir -p cluster/logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh; conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
MODELS=(Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
BS=16; case "$M" in Qwen/Qwen3-0.6B|Qwen/Qwen3-1.7B|Qwen/Qwen3-4B) BS=32;; esac
echo "task $SLURM_ARRAY_TASK_ID model $M bs $BS host $(hostname)"
python code/cluster_grid_corpus.py --model "$M" --out_root results/instruct_grid --n_sents 300 --batch_size $BS --dtype bf16 --device cuda
