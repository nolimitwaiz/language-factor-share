#!/bin/bash
#SBATCH --job-name=rmfs02-join
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=cluster_logs/rmfs02_join_%j.out
#SBATCH --error=cluster_logs/rmfs02_join_%j.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/profile02_arm_audit
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/assemble_rmfs_profile02_arm_audit.py \
  --profiles-root "$HOME/rmfs/results/profile02_arms/checkpoints" \
  --reference-profile "$HOME/rmfs/results/profile02_arms/reference/result.json" \
  --frozen-arm A \
  --frozen-arm WA-A \
  --outcomes "$HOME/rmfs/data/external/aim2_all_arm_seed_metrics.csv" \
  --output-dir "$HOME/rmfs/results/profile02_arm_audit" \
  --expected-checkpoints 81
