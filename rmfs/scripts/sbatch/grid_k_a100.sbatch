#!/bin/bash
#SBATCH --job-name=rmfs-grid
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
# no exclude list: that list dodges Pascal-GPU torch crashes and is
# irrelevant for CPU-only analysis; keeping it here starved the array
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --array=0-13
#SBATCH --output=cluster_logs/grid_%A_%a.log

# Phase 1.7: LFS-VC / LDE / K / gates / shared-prep baselines on the 14
# dump models. CPU-only analysis of saved embeddings; no model loaded.
# PROLOGUE RUNS THE UNIT BATTERY: no grid number is produced on a machine
# where the battery has not passed (estimator_notes N3, environment note).
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/runs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$HOME/rmfs/src"
python -m pytest tests -q -p no:warnings || { echo "UNIT BATTERY RED - refusing to compute grid numbers"; exit 1; }
MODELS=(Qwen3-0.6B-Base Qwen3-1.7B-Base Qwen3-4B-Base Qwen3-8B-Base \
        OLMo-2-0425-1B OLMo-2-1124-7B bloom-1b7 bloom-7b1 \
        salamandra-2b salamandra-7b EuroLLM-1.7B SmolLM2-1.7B \
        Falcon3-7B-Base Mistral-7B-v0.3)
python scripts/grid_components.py --model_tag "${MODELS[$SLURM_ARRAY_TASK_ID]}"
