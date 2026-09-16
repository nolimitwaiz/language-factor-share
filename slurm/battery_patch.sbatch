#!/bin/bash
#SBATCH --job-name=mlm-patch-a1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=cluster/logs/patch_%A_%a.log

# Addendum A1 patch run: I4b (centroid-preserving rotation) + I8b
# (global-centroid collapse) cells only, on the existing Phase-0 dumps.
# Predictions frozen in PREREG_BATTERY.md Addendum A1 (2026-07-16) BEFORE
# this submission.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

TAGS=("Qwen3-1.7B-Base" "OLMo-2-0425-1B" "bloom-1b7")
T=${TAGS[$SLURM_ARRAY_TASK_ID]}
python src/battery/score.py --model_tag "$T" --cells I4b,I8b --out_suffix a1
