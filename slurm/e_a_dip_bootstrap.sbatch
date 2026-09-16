#!/bin/bash
#SBATCH --job-name=aim1-ea-boot
#SBATCH --partition=cpu
#SBATCH --array=0-13
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/logs/ea_boot_%A_%a.out
#SBATCH --error=cluster/logs/ea_boot_%A_%a.err
set -eo pipefail
cd "$HOME/multilingual-metrics"; mkdir -p cluster/logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh; conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
MODELS=(bloom-1b7 bloom-7b1 EuroLLM-1.7B Falcon3-7B-Base Mistral-7B-v0.3 OLMo-2-0425-1B OLMo-2-1124-7B Qwen3-0.6B-Base Qwen3-1.7B-Base Qwen3-4B-Base Qwen3-8B-Base salamandra-2b salamandra-7b SmolLM2-1.7B)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; echo "task $SLURM_ARRAY_TASK_ID model $M host $(hostname)"
python src/analysis/dip_bootstrap.py --model "$M" --n_boot 2000 --n_sub 300
