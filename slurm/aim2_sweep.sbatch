#!/bin/bash
#SBATCH --job-name=mlm-sweep
#SBATCH --partition=gpu
#SBATCH --exclude=c02,c03,c04,c05,c08,c09,c10,c11,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c26,c27,octopod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=cluster/logs/sweep_%A_%a.log

# The two follow-ups ranked highest after the LFS study.
#   0-2: guard-weight sweep on arm E. The hinge at weight 1.0 produced a
#        penalty near 0.0025 against an alignment loss of order 3.4, so the
#        constraint it names was never enforced. This asks what an adequately
#        weighted guard does to tail alignment, which is unknown.
#   3:   arm G, which optimizes the tail directly. The LFS study drove LFS
#        down and watched AaR follow it down; it never tried driving AaR up.
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
SEED="${SEED:-0}"
case $SLURM_ARRAY_TASK_ID in
  0) python -m src.aim2.train --arm E --cvp_weight 10   --tag _cvp10  --steps 400 --seed "$SEED" ;;
  1) python -m src.aim2.train --arm E --cvp_weight 100  --tag _cvp100 --steps 400 --seed "$SEED" ;;
  2) python -m src.aim2.train --arm E --cvp_weight 1000 --tag _cvp1k  --steps 400 --seed "$SEED" ;;
  3) python -m src.aim2.train --arm G --cvp_weight 100  --tag _tail   --steps 400 --seed "$SEED" ;;
esac
