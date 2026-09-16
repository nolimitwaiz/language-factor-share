#!/bin/bash
#SBATCH --job-name=round3-invar
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-13%7
#SBATCH --output=cluster_logs/round3_invariance_%A_%a.out
#SBATCH --error=cluster_logs/round3_invariance_%A_%a.err

set -eo pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/round3_invariance/parts
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/rmfs/src"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

MANIFEST="$HOME/rmfs/manifests/round3_invariance_models_2026-08-24.txt"
if [[ ! -s "$MANIFEST" ]]; then
  echo "frozen model manifest is missing: $MANIFEST" >&2
  exit 1
fi
mapfile -t MODELS < "$MANIFEST"
if [[ "${#MODELS[@]}" -ne 14 ]]; then
  echo "expected 14 model entries, found ${#MODELS[@]}" >&2
  exit 1
fi
if [[ "$(printf '%s\n' "${MODELS[@]}" | sort -u | wc -l)" -ne 14 ]]; then
  echo "model manifest contains duplicates" >&2
  exit 1
fi

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
if [[ ! -d "$HOME/multilingual-metrics/results/dumps/$MODEL" ]]; then
  echo "campaign dump is missing: $MODEL" >&2
  exit 1
fi
echo "task $SLURM_ARRAY_TASK_ID -> $MODEL"
python scripts/round3_invariance.py --model "$MODEL" --n-concepts 300
