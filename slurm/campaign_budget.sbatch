#!/bin/bash
#SBATCH --job-name=mlm-camp-budget
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-41
#SBATCH --output=cluster/logs/cbudget_%A_%a.log

# Campaign budgets (prereg A2): 14 models x 3 configs.
# config 0 = selected rank, raw subspace, WITH nulls (primary)
# config 1 = fixed r=64, raw subspace, no nulls (dual-rank mitigation)
# config 2 = selected rank, centered subspace, no nulls (A2 variant)
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/budget

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

TAGS=("Qwen3-0.6B-Base" "Qwen3-1.7B-Base" "Qwen3-4B-Base" "Qwen3-8B-Base" \
      "OLMo-2-0425-1B" "OLMo-2-1124-7B" "Mistral-7B-v0.3" "bloom-1b7" \
      "bloom-7b1" "EuroLLM-1.7B" "SmolLM2-1.7B" "Falcon3-7B-Base" \
      "salamandra-2b" "salamandra-7b")
T=${TAGS[$((SLURM_ARRAY_TASK_ID / 3))]}
C=$((SLURM_ARRAY_TASK_ID % 3))

case $C in
  0) python src/analysis/budget_real.py --model_tag "$T" ;;
  1) python src/analysis/budget_real.py --model_tag "$T" --rank_mode fixed --skip_nulls ;;
  2) python src/analysis/budget_real.py --model_tag "$T" --subspace centered --skip_nulls ;;
esac
