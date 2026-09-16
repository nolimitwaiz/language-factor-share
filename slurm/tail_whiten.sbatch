#!/bin/bash
#SBATCH --job-name=mlm-tailwhiten
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-13
#SBATCH --output=cluster/logs/tw_%A_%a.log
set -o pipefail
cd "$HOME/multilingual-metrics"
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
TAGS=("Qwen3-0.6B-Base" "Qwen3-1.7B-Base" "Qwen3-4B-Base" "Qwen3-8B-Base" \
      "OLMo-2-0425-1B" "OLMo-2-1124-7B" "Mistral-7B-v0.3" "bloom-1b7" \
      "bloom-7b1" "EuroLLM-1.7B" "SmolLM2-1.7B" "Falcon3-7B-Base" \
      "salamandra-2b" "salamandra-7b")
python src/analysis/tail_whiten.py --model_tag "${TAGS[$SLURM_ARRAY_TASK_ID]}"
