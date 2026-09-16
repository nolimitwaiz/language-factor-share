#!/bin/bash
#SBATCH --job-name=round3-join
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0:30:00
#SBATCH --output=cluster_logs/round3_invariance_join_%j.out
#SBATCH --error=cluster_logs/round3_invariance_join_%j.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/assemble_round3_invariance.py \
  --parts "$HOME/rmfs/results/round3_invariance/parts" \
  --model-manifest "$HOME/rmfs/manifests/round3_invariance_models_2026-08-24.txt" \
  --output-dir "$HOME/rmfs/results/round3_invariance/assembled_2026-08-24"
