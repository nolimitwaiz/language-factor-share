#!/bin/bash
#SBATCH --job-name=mlm-attn313
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=cluster/logs/attn313_%j.log

# Attention mechanism for the cross-lingual context result.
# Predictions frozen in prereg/PREREG_ATTENTION_313.md before this ran.
# Eager attention is required to obtain attention weights, which makes the
# forward pass slower and heavier than the behavioral experiment.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/attention313

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name --format=csv,noheader || true
python code/attention_313.py --model Qwen/Qwen3-0.6B-Base --n_pairs 60
