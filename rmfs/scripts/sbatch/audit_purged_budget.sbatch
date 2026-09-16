#!/bin/bash
#SBATCH --job-name=rmfs-audit
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=cluster_logs/audit_%j.log

# Phase 0.3b purged-split audit. CPU-only linear algebra on the saved
# Qwen3-1.7B n=1500 dump (no GPU requested). Estimated wall time: 1-3 h
# (the published 34h primary was dominated by permutation nulls and the
# rank grid, both skipped here; this runs 2 arms x 20 splits x ~L langs
# of ladder fits at the published rank).
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/runs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
python scripts/audit/audit_purged_budget.py --model_tag Qwen3-1.7B-Base
