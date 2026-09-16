#!/bin/bash
#SBATCH --job-name=mlm-wlalign
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:30:00
#SBATCH --output=cluster/logs/wlalign_%j.log

# Word-level stage 0: alignment infrastructure over the twenty-language
# prototype set. No LLM representations are computed; the GPU is used only
# to run XLM-R forward passes for the neural aligner.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/wordlevel/alignments

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"     # login node root filesystem is full
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=name --format=csv,noheader || true
python -m src.wordlevel.align \
    --languages configs/wordlevel/prototype20.yaml \
    --n_sents 300 --n_train 2000
