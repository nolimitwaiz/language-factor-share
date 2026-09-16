#!/bin/bash
#SBATCH --job-name=rmfs-newrd
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --array=0-20
#SBATCH --output=cluster_logs/newrd_%A_%a.log
# A9 stage 1: layer sweep + readings for models new to the grid.
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/runs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/rmfs/src"
export OMP_NUM_THREADS=4
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs/new_models.tsv)
TAG=$(echo "$LINE" | cut -f1)
HF=$(echo "$LINE" | cut -f2)
echo "task $SLURM_ARRAY_TASK_ID -> $TAG ($HF)"
python scripts/new_model_readings.py --tag "$TAG" --hf_id "$HF"
