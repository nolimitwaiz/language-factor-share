#!/bin/bash
#SBATCH --job-name=mlm-adv
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=cluster/logs/advprobe_%j.log

# Section 3.2.3 premise test: does LFS predict what a language discriminator
# can exploit? Runs before the adversarial arm is built, because if the
# premise fails the arm should not be built on this metric.
set -o pipefail
cd "$HOME/multilingual-metrics"
mkdir -p cluster/logs results/adversarial_probe
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -m src.aim2.adversarial_probe --model Qwen/Qwen3-0.6B-Base --n_langs 24
