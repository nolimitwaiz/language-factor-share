#!/bin/bash
#SBATCH --job-name=rmfs02-arms
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=0:20:00
#SBATCH --array=0-80%20
#SBATCH --output=cluster_logs/rmfs02_arms_%A_%a.out
#SBATCH --error=cluster_logs/rmfs02_arms_%A_%a.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/profile02_arms/checkpoints
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/rmfs/src:$HOME/multilingual-metrics"
export OMP_NUM_THREADS=4

MANIFEST="$HOME/rmfs/manifests/aim2_scientific_checkpoints_2026-08-24.txt"
if [[ ! -s "$MANIFEST" ]]; then
  echo "frozen checkpoint manifest is missing: $MANIFEST" >&2
  exit 1
fi
mapfile -t CHECKPOINTS < "$MANIFEST"
if [[ "${#CHECKPOINTS[@]}" -ne 81 ]]; then
  echo "expected 81 manifest entries, found ${#CHECKPOINTS[@]}" >&2
  exit 1
fi
if [[ "$(printf '%s\n' "${CHECKPOINTS[@]}" | sort -u | wc -l)" -ne 81 ]]; then
  echo "checkpoint manifest contains duplicates" >&2
  exit 1
fi
for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ "$checkpoint" == *smoke* ]]; then
    echo "smoke checkpoint is forbidden in the scientific manifest: $checkpoint" >&2
    exit 1
  fi
  if [[ ! -d "$HOME/multilingual-metrics/results/aim2/$checkpoint" ]]; then
    echo "manifest checkpoint is missing: $checkpoint" >&2
    exit 1
  fi
done
if [[ ! -s "$HOME/rmfs/results/profile02_arms/reference/base_grid.npz" ]]; then
  echo "frozen reference grid is missing" >&2
  exit 1
fi

CHECKPOINT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> $CHECKPOINT"
python scripts/rmfs_profile_arms_v02.py --arm-ckpt "$CHECKPOINT"
