#!/bin/bash
#SBATCH --job-name=rmfs-incl
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:50:00
#SBATCH --array=0-13
#SBATCH --output=cluster_logs/incl_%A_%a.log
# A8: INCLUDE, natively authored regional exams, 30 panel languages.
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/include_native
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
MODELS=($(ls ~/multilingual-metrics/results/dumps/ | sort))
TAG="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> $TAG"
HF=$(python - << 'PY'
import yaml,os
r=yaml.safe_load(open(os.path.expanduser("~/rmfs/configs/models.yaml")))["models"]
import sys; print(r[""]["hf_id"] if "" in r else "")
PY
)
HF=$(python -c "
import yaml,os,sys
r=yaml.safe_load(open(os.path.expanduser('~/rmfs/configs/models.yaml')))['models']
print(r['$TAG']['hf_id'])")
echo "hf id: $HF"
DT=bfloat16; case "$TAG" in *bloom*) DT=float32;; esac
lm_eval --model hf \
  --model_args "pretrained=$HF,dtype=$DT,trust_remote_code=True" \
  --tasks "include_base_44_arabic,include_base_44_bengali,include_base_44_bulgarian,include_base_44_chinese,include_base_44_dutch,include_base_44_estonian,include_base_44_finnish,include_base_44_french,include_base_44_georgian,include_base_44_german,include_base_44_greek,include_base_44_hebrew,include_base_44_hindi,include_base_44_hungarian,include_base_44_indonesian,include_base_44_italian,include_base_44_japanese,include_base_44_korean,include_base_44_malay,include_base_44_persian,include_base_44_polish,include_base_44_portuguese,include_base_44_russian,include_base_44_spanish,include_base_44_tamil,include_base_44_telugu,include_base_44_turkish,include_base_44_ukrainian,include_base_44_urdu,include_base_44_vietnamese" --num_fewshot 0 --batch_size auto \
  --output_path "results/include_native/$TAG"
