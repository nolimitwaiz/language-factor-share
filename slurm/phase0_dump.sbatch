#!/bin/bash
#SBATCH --job-name=mlm-phase0
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --array=0-2
#SBATCH --output=cluster/logs/phase0_%A_%a.log

# Phase 0: prototype embedding dumps (3 models x 5 selected layers) + parity
# report against the prior grid numbers. Pinned to c24 (RTX 2080 Ti):
# Pascal nodes crash torch cu128 (no sm_61 kernels), and matching the
# original grid hardware keeps the golden-file parity comparison clean.
# BLOOM never in fp16 (bf16-trained; incident #2) -> fp32 here.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/dumps

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODELS=("Qwen/Qwen3-1.7B-Base" "allenai/OLMo-2-0425-1B" "bigscience/bloom-1b7")
DTYPES=("fp16" "fp16" "fp32")
BATCH=(16 16 8)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
D=${DTYPES[$SLURM_ARRAY_TASK_ID]}
B=${BATCH[$SLURM_ARRAY_TASK_ID]}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python src/campaign/dump_embeddings.py --model "$M" --n_sents 300 \
    --dtype "$D" --batch_size "$B" --device cuda
echo "=== PARITY REPORT ==="
python src/campaign/dump_embeddings.py --model "$M" --parity_only
