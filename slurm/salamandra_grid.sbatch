#!/bin/bash
#SBATCH --job-name=mlm-sala-grid
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=cluster/logs/sala_grid_%A_%a.log

# Salamandra grid runs (campaign G5 holdouts; predictions frozen in
# PREREG_BATTERY.md Addendum A2 BEFORE this submission). fp32: Salamandra is
# bf16-trained (Llama-arch, massive activations possible) and 2080Ti has no
# bf16 — the BLOOM incident rule applies until fp16 is proven safe.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/grid

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODELS=("BSC-LT/salamandra-2b" "BSC-LT/salamandra-7b")
BATCH=(8 4)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
B=${BATCH[$SLURM_ARRAY_TASK_ID]}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python code/cluster_grid.py --model "$M" --n_sents 300 --batch_size "$B" --dtype fp32
