#!/bin/bash
#SBATCH --job-name=mlm-batt1500
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=cluster/logs/b1500_%A_%a.log
# Battery at campaign n=1500 (prototype models): scores the A2 M5-overfit
# prediction inside the battery and re-validates calibration at scale.
set -o pipefail
cd "$HOME/multilingual-metrics"
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
TAGS=("Qwen3-1.7B-Base" "OLMo-2-0425-1B" "bloom-1b7")
python src/battery/score.py --model_tag "${TAGS[$SLURM_ARRAY_TASK_ID]}" --out_suffix n1500
