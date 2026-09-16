#!/bin/bash
#SBATCH --job-name=rmfs-armvc
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:45:00
#SBATCH --array=0-75
#SBATCH --output=cluster_logs/armvc_%A_%a.log
# REML components per arm checkpoint (tasks 0-74) + the frozen base
# reference (task 75) at the identical n=300 protocol. Feeds q_L and
# C_pres for intervention-mode RMFS assembly. Embed-only: no NLL passes.
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
if [ "$SLURM_ARRAY_TASK_ID" -eq 75 ]; then
  echo "task 75 -> base reference Qwen3-0.6B-Base"
  python scripts/arm_components.py --model_tag Qwen3-0.6B-Base
else
  echo "task $SLURM_ARRAY_TASK_ID -> ${CKPTS[$SLURM_ARRAY_TASK_ID]}"
  python scripts/arm_components.py --arm_ckpt "${CKPTS[$SLURM_ARRAY_TASK_ID]}"
fi
