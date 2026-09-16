#!/bin/bash
#SBATCH --job-name=mlm-evsweep
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=cluster/logs/evsweep_%j.log

# Downstream evaluation of the study arms. Answers the questions the
# training loop cannot: did anything actually improve, on sentences and
# languages no arm trained on.
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
python -m src.aim2.evaluate --seeds ${SEEDS:-0} --layer 8 --arms A B E_cvp10 E_cvp100 E_cvp1k G_tail
python -m src.aim2.context_use --seeds ${SEEDS:-0} --n_pairs 200 --arms A B E_cvp10 E_cvp100 E_cvp1k G_tail
