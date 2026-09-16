#!/bin/bash
#SBATCH --job-name=aim1-basis
#SBATCH --partition=cpu
#SBATCH --array=0-13
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=cluster_logs/aim1_basis_%A_%a.out
#SBATCH --error=cluster_logs/aim1_basis_%A_%a.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1

MODEL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" manifests/round3_invariance_models_2026-08-24.txt)
test -n "$MODEL"

python scripts/axis_basis_sensitivity.py \
  --model "$MODEL" \
  --dumps "$HOME/multilingual-metrics/results/dumps" \
  --source-manifest "$HOME/rmfs/results/round3_anisotropy_axes/parts/${MODEL}.manifest.json" \
  --output-root "$HOME/rmfs/results/round3/axis_basis_sensitivity_2026-08-24/parts"
