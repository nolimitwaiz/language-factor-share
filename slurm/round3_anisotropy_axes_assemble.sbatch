#!/bin/bash
#SBATCH --job-name=round3-axesjoin
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=cluster_logs/round3_axes_join_%j.out
#SBATCH --error=cluster_logs/round3_axes_join_%j.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/assemble_round3_anisotropy_axes.py \
  --parts "$HOME/rmfs/results/round3_anisotropy_axes/parts" \
  --models "$HOME/rmfs/manifests/round3_invariance_models_2026-08-24.txt" \
  --output-dir "$HOME/rmfs/results/round3_anisotropy_axes/assembled_2026-08-24"
