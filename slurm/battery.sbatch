#!/bin/bash
#SBATCH --job-name=mlm-battery
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --array=0-2
#SBATCH --output=cluster/logs/battery_%A_%a.log

# Battery (frozen prereg 2026-07-15) + real-data misalignment budget on the
# Phase-0 dumps. CPU-only: analyzes saved embeddings, no model forward
# passes. One array task per prototype model.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/battery results/budget

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

TAGS=("Qwen3-1.7B-Base" "OLMo-2-0425-1B" "bloom-1b7")
T=${TAGS[$SLURM_ARRAY_TASK_ID]}

echo "=== BATTERY: $T ==="
python src/battery/score.py --model_tag "$T"
echo "=== REAL-DATA BUDGET: $T ==="
python src/analysis/budget_real.py --model_tag "$T"
