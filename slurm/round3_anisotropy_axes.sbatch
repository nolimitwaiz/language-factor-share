#!/bin/bash
#SBATCH --job-name=round3-axes
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-13%4
#SBATCH --output=cluster_logs/round3_axes_%A_%a.out
#SBATCH --error=cluster_logs/round3_axes_%A_%a.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/round3_anisotropy_axes/parts
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/rmfs/src"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

MANIFEST="$HOME/rmfs/manifests/round3_invariance_models_2026-08-24.txt"
mapfile -t MODELS < "$MANIFEST"
if [[ "${#MODELS[@]}" -ne 14 ]]; then
  echo "expected 14 frozen models, found ${#MODELS[@]}" >&2
  exit 1
fi
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
python scripts/round3_anisotropy_axes.py --model "$MODEL" --n-concepts 300
