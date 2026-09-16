#!/bin/bash
#SBATCH --job-name=rmfs-belarm
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --array=0-74
#SBATCH --output=cluster_logs/belarm_%A_%a.log
# A5 arms downstream exam: Belebele 0-shot on every trained checkpoint,
# 41 declared panel-language tasks, protocol-matched to the campaign
# healthy-model runs (lm-eval-harness, bf16, trust_remote_code).
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/belebele_arms
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
CKPTS=($(ls -d ~/multilingual-metrics/results/aim2/ckpt_* | xargs -n1 basename | sort))
ARM="${CKPTS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> $ARM"
TASKS="belebele_amh_Ethi,belebele_arb_Arab,belebele_ben_Beng,belebele_bul_Cyrl,belebele_ces_Latn,belebele_deu_Latn,belebele_ell_Grek,belebele_eng_Latn,belebele_est_Latn,belebele_fin_Latn,belebele_fra_Latn,belebele_heb_Hebr,belebele_hin_Deva,belebele_hun_Latn,belebele_ind_Latn,belebele_ita_Latn,belebele_jpn_Jpan,belebele_kat_Geor,belebele_khm_Khmr,belebele_kor_Hang,belebele_mar_Deva,belebele_mya_Mymr,belebele_nld_Latn,belebele_pes_Arab,belebele_pol_Latn,belebele_por_Latn,belebele_ron_Latn,belebele_rus_Cyrl,belebele_spa_Latn,belebele_swe_Latn,belebele_swh_Latn,belebele_tam_Taml,belebele_tel_Telu,belebele_tha_Thai,belebele_tur_Latn,belebele_ukr_Cyrl,belebele_urd_Arab,belebele_vie_Latn,belebele_yor_Latn,belebele_zho_Hans,belebele_zsm_Latn"
lm_eval --model hf \
  --model_args "pretrained=$HOME/multilingual-metrics/results/aim2/$ARM,dtype=bfloat16,trust_remote_code=True" \
  --tasks "$TASKS" --num_fewshot 0 --batch_size auto \
  --output_path "results/belebele_arms/$ARM"
