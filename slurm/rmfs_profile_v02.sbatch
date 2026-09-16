#!/bin/bash
#SBATCH --job-name=mlm-rmfs02
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=cluster/logs/rmfs02_%j.log

cd /home/wkhan12/multilingual-metrics
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
set -euo pipefail
mkdir -p cluster/logs results

python src/analysis/rmfs_profile_campaign.py \
  --dumps results/dumps \
  --selector-results results/aim3_layer_selector_results.json \
  --layer-source selector \
  --output results/rmfs_profile_v02_14models.json
