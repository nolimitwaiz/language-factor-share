#!/bin/bash
#SBATCH --job-name=rmfs-layers
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-13
#SBATCH --output=cluster_logs/layers_%A_%a.log
# Aim-1 layerwise profiles over existing dumps (CPU-only; a100 backfill).
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/runs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/rmfs/src"
export OMP_NUM_THREADS=16
python -m pytest tests -q -p no:warnings || { echo "BATTERY RED"; exit 1; }
MODELS=($(ls ~/multilingual-metrics/results/dumps/ | sort))
echo "task $SLURM_ARRAY_TASK_ID -> ${MODELS[$SLURM_ARRAY_TASK_ID]}"
python scripts/layer_profiles.py --model_tag "${MODELS[$SLURM_ARRAY_TASK_ID]}"
