#!/bin/bash
#SBATCH --job-name=rmfs-tbase
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-3
#SBATCH --output=cluster_logs/tbase_%A_%a.log
# Behavioral T (A4 injection), full protocol: 40 languages x 60 held-out
# pairs, per base model. Llama/granite/Yi run separately with in-job rung
# selection (A3 section 4).
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/runs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/rmfs/src"
export OMP_NUM_THREADS=4
python -m pytest tests -q -p no:warnings || { echo "BATTERY RED"; exit 1; }
MODELS=(Qwen3-0.6B-Base Qwen3-4B-Base OLMo-2-0425-1B bloom-1b7)
python scripts/t_extract.py --model_tag "${MODELS[$SLURM_ARRAY_TASK_ID]}"
