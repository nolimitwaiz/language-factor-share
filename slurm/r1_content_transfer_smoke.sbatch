#!/bin/bash
#SBATCH --job-name=aim1-r1-smoke
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=cluster_logs/aim1_r1_smoke_%j.out
#SBATCH --error=cluster_logs/aim1_r1_smoke_%j.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python scripts/r1_content_transfer.py \
  --tag SmolLM2-360M \
  --model-lock "$HOME/rmfs/manifests/R1_A9_MODEL_REVISION_LOCK_2026-08-24.tsv" \
  --pairs "$HOME/rmfs/results/round3/r1_content_pairs_2026-08-24/pairs.csv" \
  --data-manifest "$HOME/rmfs/results/round3/r1_content_pairs_2026-08-24/manifest.json" \
  --ntrex "$HOME/multilingual-metrics/data/NTREX/NTREX-128" \
  --output-root "$HOME/rmfs/results/round3/r1_content_transfer_2026-08-24" \
  --smoke
