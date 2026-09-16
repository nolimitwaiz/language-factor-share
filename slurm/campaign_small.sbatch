#!/bin/bash
#SBATCH --job-name=mlm-camp-s
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --array=0-7
#SBATCH --output=cluster/logs/camp_s_%A_%a.log

# Campaign re-embed (prereg A2, frozen 2026-07-17): small models on 2080Ti.
# n=1500, 5 selected layers, fp16 storage after fp32 pooling, per-sentence
# margins saved. fp32 for bf16-trained models (BLOOM rule).
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/dumps

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODELS=("Qwen/Qwen3-0.6B-Base" "Qwen/Qwen3-1.7B-Base" "Qwen/Qwen3-4B-Base" \
        "allenai/OLMo-2-0425-1B" "utter-project/EuroLLM-1.7B" \
        "HuggingFaceTB/SmolLM2-1.7B" "bigscience/bloom-1b7" "BSC-LT/salamandra-2b")
DTYPES=("fp16" "fp16" "fp16" "fp16" "fp16" "fp16" "fp32" "fp32")
BATCH=(16 16 12 16 16 16 8 8)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
D=${DTYPES[$SLURM_ARRAY_TASK_ID]}
B=${BATCH[$SLURM_ARRAY_TASK_ID]}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python src/campaign/dump_embeddings.py --model "$M" --n_sents 1500 \
    --dtype "$D" --batch_size "$B" --device cuda --save_margins
