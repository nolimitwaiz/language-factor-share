#!/bin/bash
#SBATCH --job-name=round3-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=cluster_logs/round3_empirical_summary_%j.out
#SBATCH --error=cluster_logs/round3_empirical_summary_%j.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/summarize_round3_empirical.py \
  --sweep "$HOME/rmfs/results/round3_invariance/assembled_2026-08-24/sweep.csv" \
  --output-dir "$HOME/rmfs/results/round3_invariance/empirical_summary_2026-08-24"
