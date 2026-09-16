#!/bin/bash
#SBATCH --job-name=aim1-eb-flores
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --array=0-16%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=02:30:00
#SBATCH --output=cluster/logs/eb_flores_%A_%a.out
#SBATCH --error=cluster/logs/eb_flores_%A_%a.err
set -eo pipefail
cd "$HOME/multilingual-metrics"; mkdir -p cluster/logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh; conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
MODELS=(Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base Qwen/Qwen3-8B-Base allenai/OLMo-2-0425-1B allenai/OLMo-2-1124-7B mistralai/Mistral-7B-v0.3 bigscience/bloom-1b7 bigscience/bloom-7b1 utter-project/EuroLLM-1.7B HuggingFaceTB/SmolLM2-1.7B tiiuae/Falcon3-7B-Base BSC-LT/salamandra-2b BSC-LT/salamandra-7b ibm-granite/granite-3.1-8b-base 01-ai/Yi-1.5-9B meta-llama/Llama-3.1-8B)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
DT=bf16; BS=16
case "$M" in
  bigscience/bloom-1b7) DT=fp32; BS=32;;
  Qwen/Qwen3-0.6B-Base|Qwen/Qwen3-1.7B-Base|Qwen/Qwen3-4B-Base|allenai/OLMo-2-0425-1B|utter-project/EuroLLM-1.7B|HuggingFaceTB/SmolLM2-1.7B|BSC-LT/salamandra-2b) BS=32;;
esac
echo "task $SLURM_ARRAY_TASK_ID model $M dtype $DT bs $BS host $(hostname)"
python code/cluster_grid_corpus.py --model "$M" \
  --data_dir data/FLORES200/devtest_ntrex_layout --src_file flores200-devtest.eng.txt --ref_prefix flores200-devtest. \
  --out_root results/flores_grid --n_sents 300 --batch_size $BS --dtype $DT --device cuda
