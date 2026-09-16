#!/bin/bash
#SBATCH --job-name=aim2_w0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-13
#SBATCH --output=cluster/logs/aim2_w0_%A_%a.out
#SBATCH --error=cluster/logs/aim2_w0_%A_%a.err
# W0: offline offset/scale removal on the saved dumps. Protocol frozen 2026-09-07:
# prereg/addenda/AIM2_W0_OFFSET_OFFLINE_PROTOCOL_2026-09-07.md. Zero GPU.
set -euo pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs
MODELS=(Qwen3-0.6B-Base Qwen3-1.7B-Base Qwen3-4B-Base Qwen3-8B-Base OLMo-2-0425-1B OLMo-2-1124-7B Mistral-7B-v0.3 bloom-1b7 bloom-7b1 EuroLLM-1.7B SmolLM2-1.7B Falcon3-7B-Base salamandra-2b salamandra-7b)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "[$(date)] W0 start $M on $(hostname)"
"$HOME/venvs/mlm/bin/python" src/aim2/offset_offline.py --model "$M" --n_boot 300
echo "[$(date)] W0 done $M"
