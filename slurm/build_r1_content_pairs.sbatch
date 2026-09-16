#!/bin/bash
#SBATCH --job-name=aim1-r1-pairs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=cluster_logs/aim1_r1_pairs_%j.out
#SBATCH --error=cluster_logs/aim1_r1_pairs_%j.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/build_r1_content_pairs.py \
  --ntrex "$HOME/multilingual-metrics/data/NTREX/NTREX-128" \
  --document-ids "$HOME/multilingual-metrics/data/NTREX/DOCUMENT_IDS.tsv" \
  --output-dir "$HOME/rmfs/results/round3/r1_content_pairs_2026-08-24" \
  --n-pairs 100
