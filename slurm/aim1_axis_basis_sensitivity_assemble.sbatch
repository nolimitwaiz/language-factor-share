#!/bin/bash
#SBATCH --job-name=aim1-basis-join
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=cluster_logs/aim1_basis_join_%j.out
#SBATCH --error=cluster_logs/aim1_basis_join_%j.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/assemble_axis_basis_sensitivity.py \
  --parts "$HOME/rmfs/results/round3/axis_basis_sensitivity_2026-08-24/parts" \
  --models "$HOME/rmfs/manifests/round3_invariance_models_2026-08-24.txt" \
  --output-dir "$HOME/rmfs/results/round3/axis_basis_sensitivity_2026-08-24/assembled"
