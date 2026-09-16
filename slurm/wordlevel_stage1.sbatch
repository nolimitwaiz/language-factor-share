#!/bin/bash
#SBATCH --job-name=mlm-wl1
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --array=0-1
#SBATCH --output=cluster/logs/wl1_%A_%a.log

# Word-level stage 1. Gates and predictions frozen in
# prereg/PREREG_WORDLEVEL.md before any word-level state was extracted.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/wordlevel/stage1
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
POINTS=("primary" "secondary")
python -m src.wordlevel.stage1 --model Qwen/Qwen3-1.7B-Base \
    --operating_point "${POINTS[$SLURM_ARRAY_TASK_ID]}"
