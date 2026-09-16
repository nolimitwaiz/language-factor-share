#!/bin/bash
#SBATCH --job-name=rmfs-transl
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --array=0-13
#SBATCH --output=cluster_logs/transl_%A_%a.log
# A7: native versus translated text, WMT19 both directions.
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/runs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export SACREBLEU="$HOME/.sacrebleu"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/rmfs/src"
export OMP_NUM_THREADS=4
python -m pytest tests -q -p no:warnings || { echo "BATTERY RED"; exit 1; }
MODELS=($(ls ~/multilingual-metrics/results/dumps/ | sort))
echo "task $SLURM_ARRAY_TASK_ID -> ${MODELS[$SLURM_ARRAY_TASK_ID]}"
python scripts/translationese.py --model_tag "${MODELS[$SLURM_ARRAY_TASK_ID]}"
