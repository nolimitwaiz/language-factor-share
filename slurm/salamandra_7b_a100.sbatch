#!/bin/bash
#SBATCH --job-name=mlm-sala7b
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/sala7b_%j.log

# salamandra-7b grid: fp32 does not fit a 2080Ti (OOM, job 1691514_1);
# A100 + bf16 (native — Salamandra is bf16-trained, same as the bloom-7b1
# routing in the original grid). Short walltime -> backfill-friendly.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/grid

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python code/cluster_grid.py --model BSC-LT/salamandra-7b --n_sents 300 \
    --batch_size 16 --dtype bf16
