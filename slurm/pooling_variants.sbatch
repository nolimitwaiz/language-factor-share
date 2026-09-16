#!/bin/bash
#SBATCH --job-name=mlm-pool
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-4
#SBATCH --output=cluster/logs/pool_%A_%a.log

# Fertility and pooling robustness. Decision rule frozen in
# docs/FERTILITY_ROBUSTNESS_PLAN.md before this ran. Five models spanning
# recipe classes; Qwen3 0.6B and 8B share a tokenizer and serve as the
# within-family control where fertility is identical by construction.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/pooling_variants
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
MODELS=("Qwen/Qwen3-0.6B-Base" "Qwen/Qwen3-8B-Base" "allenai/OLMo-2-0425-1B" \
        "bigscience/bloom-1b7" "BSC-LT/salamandra-2b")
BATCH=(16 8 16 8 8)
python -m src.analysis.pooling_variants \
    --model "${MODELS[$SLURM_ARRAY_TASK_ID]}" \
    --batch_size "${BATCH[$SLURM_ARRAY_TASK_ID]}"
