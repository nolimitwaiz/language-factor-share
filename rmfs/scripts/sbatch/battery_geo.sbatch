#!/bin/bash
#SBATCH --job-name=rmfs-batgeo
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=cluster_logs/batgeo_%j.log
# S2 battery, geometric components: 14 injected worlds + base + 3
# sentence-resample floor worlds through the untouched grid pipeline.
# CPU-only; a100acct short-walltime backfill routing (CLAUDE.md policy).
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/runs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/rmfs/src"
export OMP_NUM_THREADS=1
python -m pytest tests -q -p no:warnings || { echo "BATTERY RED"; exit 1; }
python scripts/battery_geo.py --workers 20
