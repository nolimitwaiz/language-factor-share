#!/bin/bash
#SBATCH --job-name=aim1-r1-analyze
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=cluster_logs/aim1_r1_analyze_%j.out
#SBATCH --error=cluster_logs/aim1_r1_analyze_%j.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
: "${REPAIR_ARRAY_JOB_ID:?submit with --export=ALL,REPAIR_ARRAY_JOB_ID=<jobid>}"
: "${FINAL_REPAIR_ARRAY_JOB_ID:?submit with --export=ALL,FINAL_REPAIR_ARRAY_JOB_ID=<jobid>}"

python scripts/assemble_analyze_r1_content_transfer.py \
  --original-array-job-id 1761878 \
  --repair-array-job-id "$REPAIR_ARRAY_JOB_ID" \
  --final-repair-array-job-id "$FINAL_REPAIR_ARRAY_JOB_ID" \
  --model-lock "$HOME/rmfs/manifests/R1_A9_MODEL_REVISION_LOCK_2026-08-24.tsv" \
  --pairs "$HOME/rmfs/results/round3/r1_content_pairs_2026-08-24/pairs.csv" \
  --data-manifest "$HOME/rmfs/results/round3/r1_content_pairs_2026-08-24/manifest.json" \
  --parts-root "$HOME/rmfs/results/round3/r1_content_transfer_2026-08-24/parts" \
  --expanded-panel "$HOME/rmfs/results/tables/expanded_panel.csv" \
  --output "$HOME/rmfs/results/round3/r1_content_transfer_analysis_2026-08-24" \
  --n-boot 2000
