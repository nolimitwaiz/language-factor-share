#!/bin/bash
#SBATCH --job-name=rmfs-tmore
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --array=0-9
#SBATCH --output=cluster_logs/tmore_%A_%a.log
# T extension to the remaining 10 grid models (same-support MEXA showdown).
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
MODELS=(Qwen3-1.7B-Base Qwen3-8B-Base OLMo-2-1124-7B bloom-7b1 EuroLLM-1.7B SmolLM2-1.7B Falcon3-7B-Base Mistral-7B-v0.3 salamandra-2b salamandra-7b)
echo "task $SLURM_ARRAY_TASK_ID -> ${MODELS[$SLURM_ARRAY_TASK_ID]}"
python scripts/t_extract.py --model_tag "${MODELS[$SLURM_ARRAY_TASK_ID]}"
