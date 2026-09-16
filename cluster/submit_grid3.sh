#!/bin/bash
#SBATCH --job-name=mlm-grid3
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=15:00:00
#SBATCH --output=cluster/logs/grid3_%j.log

# Hold 3 A100s for the full window; 3 parallel workers, one GPU each,
# each runs its model subset sequentially (balanced by param count).
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/grid

source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

# model indices into cluster_grid.MODELS, per GPU (size-balanced)
GPU0="3 6 4 0"     # Qwen3-8B, Mistral-7B, OLMo-2-1B, Qwen3-0.6B
GPU1="10 5 7"      # EuroLLM-9B, OLMo-2-7B, bloom-1b7
GPU2="8 2 1 9"     # bloom-7b1, Qwen3-4B, Qwen3-1.7B, EuroLLM-1.7B

run_worker () {
    local gpu=$1; shift
    for idx in "$@"; do
        echo "[gpu$gpu] starting model_idx=$idx $(date)"
        CUDA_VISIBLE_DEVICES=$gpu python code/cluster_grid.py \
            --model_idx "$idx" --n_sents 300 --batch_size 48 \
            >> "cluster/logs/worker_gpu${gpu}.log" 2>&1 \
            || echo "[gpu$gpu] model_idx=$idx FAILED (continuing)"
        echo "[gpu$gpu] finished model_idx=$idx $(date)"
    done
}

run_worker 0 $GPU0 &
run_worker 1 $GPU1 &
run_worker 2 $GPU2 &
wait
echo "[all done] $(date)"
