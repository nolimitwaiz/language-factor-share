#!/bin/bash
#SBATCH --job-name=aim1-r0-analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=cluster_logs/aim1_r0_analysis_%j.out
#SBATCH --error=cluster_logs/aim1_r0_analysis_%j.err

set -eo pipefail

cd "$HOME/rmfs"
mkdir -p cluster_logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/analyze_aim1_profile_axes.py \
  --profile-dir "$HOME/rmfs/results/profile02_arm_audit" \
  --axis-dir "$HOME/rmfs/results/round3_anisotropy_axes/assembled_2026-08-24" \
  --output-dir "$HOME/rmfs/results/round3/aim1_profile_axes_analysis_2026-08-24"
