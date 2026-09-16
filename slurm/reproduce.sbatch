#!/bin/bash
#SBATCH --job-name=mlm-reproduce
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=0:45:00
#SBATCH --output=cluster/logs/reproduce_%j.log

# S1 determinism gate. Runs the pinned reproduction configuration twice in
# separate processes on the same GPU, then compares every reported quantity.
# Two runs must agree to the tolerance declared in the config file.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/reproduction

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# Required for deterministic cuBLAS reductions; must be set before CUDA init.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CFG=configs/reproduction/qwen3_0.6b_ntrex_small.yaml
nvidia-smi --query-gpu=name --format=csv,noheader || true

echo "=== RUN A ==="
python experiments/reproduce_lfs.py --config "$CFG" \
    --out results/reproduction/run_a.json --figure
echo "=== RUN B ==="
python experiments/reproduce_lfs.py --config "$CFG" \
    --out results/reproduction/run_b.json

echo "=== DETERMINISM GATE ==="
python experiments/compare_runs.py \
    results/reproduction/run_a.json results/reproduction/run_b.json
