#!/bin/bash
#SBATCH --job-name=mlm-wasmoke
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output=cluster/logs/wasmoke_%j.log
set -o pipefail
cd "$HOME/multilingual-metrics"
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
for A in WA-C WA-D WA-E; do
  echo "===== smoke $A"
  python -m src.aim2.train_wordalign --arm $A --steps 4 --log_every 1 --seed 0 --tag _SMOKE
done
