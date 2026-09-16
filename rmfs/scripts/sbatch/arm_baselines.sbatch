#!/bin/bash
#SBATCH --job-name=rmfs-armbl
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:20:00
#SBATCH --array=0-74
#SBATCH --output=cluster_logs/armbl_%A_%a.log
# A5: per-arm shared-prep MEXA/AaR at the grid protocol.
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
CKPTS=($(ls -d ~/multilingual-metrics/results/aim2/ckpt_* | xargs -n1 basename | sort))
python scripts/arm_baselines.py --arm_ckpt "${CKPTS[$SLURM_ARRAY_TASK_ID]}"
