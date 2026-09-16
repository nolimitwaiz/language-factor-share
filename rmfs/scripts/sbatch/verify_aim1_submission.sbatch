#!/bin/bash
#SBATCH --job-name=aim1_verify
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/home/wkhan12/rmfs/results/logs/aim1_verify_%j.out
#SBATCH --error=/home/wkhan12/rmfs/results/logs/aim1_verify_%j.err

set -euo pipefail

cd /home/wkhan12/rmfs
source /home/wkhan12/venvs/mlm/bin/activate

export PYTHONPATH=/home/wkhan12/rmfs/src
export MULTILINGUAL_METRICS_ROOT=/home/wkhan12/multilingual-metrics
export RMFS_DEFLATION_ROOT=/home/wkhan12/rmfs/data/external
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p /home/wkhan12/rmfs/results/logs

echo "AIM1_VERIFICATION_START"
date --iso-8601=seconds
hostname
python --version

echo "RMFS_UNIT_TESTS"
python -m pytest -q

echo "SAME_SUPPORT_LFS_RMFS"
python scripts/compare_lfs_rmfs_submission.py

echo "EXPANDED_VALIDITY_REBUILD"
python scripts/expanded_validity.py

echo "OUTPUT_HASHES"
sha256sum \
  results/submission_comparison/manifest.json \
  results/submission_comparison/same_support_estimates.csv \
  results/submission_comparison/paired_differences.csv \
  results/tables/expanded_panel.csv \
  results/tables/expanded_validity.txt

date --iso-8601=seconds
echo "AIM1_VERIFICATION_COMPLETE"
