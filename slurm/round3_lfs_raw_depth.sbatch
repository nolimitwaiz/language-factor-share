#!/bin/bash
#SBATCH --job-name=round3-lfsfix
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=cluster_logs/round3_lfs_raw_depth_%j.out
#SBATCH --error=cluster_logs/round3_lfs_raw_depth_%j.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/round3_lfs_raw_depth.py \
  --grid-results "$HOME/multilingual-metrics/results/grid" \
  --models "$HOME/rmfs/manifests/round3_invariance_models_2026-08-24.txt" \
  --report-root "$HOME/multilingual-metrics" \
  --output-dir "$HOME/rmfs/results/round3_lfs_raw_depth_2026-08-24"
