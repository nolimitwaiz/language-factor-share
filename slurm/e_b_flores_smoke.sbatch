#!/bin/bash
#SBATCH --job-name=aim1-eb-smoke
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/eb_smoke_%j.out
#SBATCH --error=cluster/logs/eb_smoke_%j.err
set -eo pipefail
cd "$HOME/multilingual-metrics"; mkdir -p cluster/logs
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh; conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python code/cluster_grid_corpus.py --model Qwen/Qwen3-0.6B-Base \
  --data_dir data/FLORES200/devtest_ntrex_layout --src_file flores200-devtest.eng.txt --ref_prefix flores200-devtest. \
  --out_root results/flores_smoke --n_sents 50 --max_langs 8 --batch_size 32 --dtype bf16 --device cuda
python - <<'PY'
import json, math
m = json.load(open("results/flores_smoke/Qwen3-0.6B-Base/metrics.json"))
prof = m["lfs_profile"]; assert all(math.isfinite(v) for v in prof) and len(prof) == 29, prof
assert m["n_sents"] == 50 and len(m["langs"]) == 9, (m["n_sents"], len(m["langs"]))
print("SMOKE OK", [round(v, 3) for v in prof[:5]], "...", m["dip_depth"])
PY
