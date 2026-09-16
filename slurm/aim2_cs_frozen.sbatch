#!/bin/bash
#SBATCH --job-name=mlm-csfz
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=cluster/logs/csfrozen_%j.log

# Backfill the frozen reference the code-switch evaluation skipped: CS-A did
# not match the hardcoded 'A' check, so criterion 6 and P5 lost their
# baseline. The fifteen trained arm-seed points are valid and are not re-run;
# only the untrained model is evaluated here and merged in.
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
python -m src.aim2.evaluate    --seeds 0 --layer 8 --arms CS-A
python -m src.aim2.context_use --seeds 0 --n_pairs 200 --arms CS-A
