#!/bin/bash
#SBATCH --job-name=mlm-nulls
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --array=0-3
#SBATCH --output=cluster/logs/nulls_%A_%a.log

# S2 estimator checks on saved campaign representations: monolingual null
# and size invariance. CPU only, no model forward passes, nothing new
# extracted.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/estimator_nulls
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
TAGS=("Qwen3-1.7B-Base" "Qwen3-8B-Base" "OLMo-2-0425-1B" "bloom-1b7")
python -m src.analysis.estimator_nulls --model_tag "${TAGS[$SLURM_ARRAY_TASK_ID]}"
