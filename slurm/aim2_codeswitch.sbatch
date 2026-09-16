#!/bin/bash
#SBATCH --job-name=mlm-cs
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --array=0-4
#SBATCH --output=cluster/logs/cs_%A_%a.log

# Code-switched training, proposal section 3.2.1.
# Design, predictions and decision rules frozen in
# prereg/PREREG_CODESWITCH.md before this ran.
# CS-A is the frozen reference and needs no job.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/aim2
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
ARMS=(CS-B CS-W10 CS-W25 CS-W50 CS-P25)
SEED="${SEED:-0}"
python -m src.aim2.train_codeswitch --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}" \
  --steps ${STEPS:-400} --seed "$SEED"
