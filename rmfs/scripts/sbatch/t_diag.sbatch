#!/bin/bash
#SBATCH --job-name=rmfs-tdiag
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=cluster_logs/tdiag_%j.log
# T-extraction smoke: 4 languages, 8 pairs, base model. Validates the hook
# substitution, LOO consensus, and reverse maps end-to-end on real GPU
# before the full arrays (A3 binding order: battery green first).
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/runs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/rmfs/src"
export OMP_NUM_THREADS=4
python -m pytest tests -q -p no:warnings || { echo "BATTERY RED"; exit 1; }
python scripts/t_extract.py --model_tag Qwen3-0.6B-Base --langs 2 --n_pairs 4 --diagnostic
