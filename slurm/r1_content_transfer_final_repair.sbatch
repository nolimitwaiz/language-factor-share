#!/bin/bash
#SBATCH --job-name=aim1-r1-final-fix
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --array=10,18%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=cluster_logs/aim1_r1_final_repair_%A_%a.out
#SBATCH --error=cluster_logs/aim1_r1_final_repair_%A_%a.err

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

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" manifests/R1_A9_MODEL_REVISION_LOCK_2026-08-24.tsv)
TAG=$(printf "%s" "$LINE" | cut -f1)
case "$TAG" in
  Yi-1.5-9B|Mistral-Nemo-Base-2407) ;;
  *) exit 2 ;;
esac

python scripts/r1_content_transfer.py \
  --tag "$TAG" \
  --model-lock "$HOME/rmfs/manifests/R1_A9_MODEL_REVISION_LOCK_2026-08-24.tsv" \
  --pairs "$HOME/rmfs/results/round3/r1_content_pairs_2026-08-24/pairs.csv" \
  --data-manifest "$HOME/rmfs/results/round3/r1_content_pairs_2026-08-24/manifest.json" \
  --ntrex "$HOME/multilingual-metrics/data/NTREX/NTREX-128" \
  --output-root "$HOME/rmfs/results/round3/r1_content_transfer_2026-08-24"
