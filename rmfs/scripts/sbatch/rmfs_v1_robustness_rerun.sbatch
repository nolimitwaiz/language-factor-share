#!/bin/bash
#SBATCH --job-name=rmfs-v1-audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=cluster_logs/rmfs_v1_audit_%j.out
#SBATCH --error=cluster_logs/rmfs_v1_audit_%j.err

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MULTILINGUAL_METRICS_ROOT="$HOME/multilingual-metrics"
export RMFS_DEFLATION_ROOT="$HOME/rmfs/data/external"
export PYTHONPATH="$HOME/rmfs/src:$HOME/multilingual-metrics"

source "$HOME/venvs/mlm/bin/activate"
cd "$HOME/rmfs"
mkdir -p cluster_logs results/rmfs_v1_robustness
python -m pytest tests -q -p no:warnings
python scripts/rmfs_v1_robustness_audit.py --n-boot 2000

