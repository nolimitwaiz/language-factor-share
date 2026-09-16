#!/bin/bash
#SBATCH --job-name=lfs_claim_audit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/home/wkhan12/rmfs/results/logs/lfs_claim_audit_%j.out
#SBATCH --error=/home/wkhan12/rmfs/results/logs/lfs_claim_audit_%j.err

set -euo pipefail

cd /home/wkhan12/rmfs
source /home/wkhan12/venvs/mlm/bin/activate

export PYTHONPATH=/home/wkhan12/rmfs/src
export MULTILINGUAL_METRICS_ROOT=/home/wkhan12/multilingual-metrics
export RMFS_DEFLATION_ROOT=/home/wkhan12/rmfs/data/external
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/audit_lfs_headline_claims.py

sha256sum \
  results/tables/lfs_headline_audit.json \
  results/tables/lfs_depth_profile_audit.csv \
  results/tables/lfs_hub_audit.csv
