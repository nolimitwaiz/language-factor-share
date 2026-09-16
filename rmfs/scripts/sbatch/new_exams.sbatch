#!/bin/bash
#SBATCH --job-name=rmfs-newex
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --array=0-18
#SBATCH --output=cluster_logs/newex_%A_%a.log
# A9 stage 2: both exams for models new to the grid.
set -o pipefail
cd "$HOME/rmfs"
mkdir -p cluster_logs results/belebele_arms results/include_native
source /opt/slurm_files/software/conda24/etc/profile.d/conda.sh
conda activate llm-experiments
source "$HOME/venvs/mlm/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp_build"
export PYTHONUNBUFFERED=1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" configs/new_models_ok.tsv)
TAG=$(echo "$LINE" | cut -f1)
HF=$(echo "$LINE" | cut -f2)
echo "task $SLURM_ARRAY_TASK_ID -> $TAG ($HF)"
lm_eval --model hf --model_args "pretrained=$HF,dtype=bfloat16,trust_remote_code=True" \
  --tasks "belebele_amh_Ethi,belebele_arb_Arab,belebele_ben_Beng,belebele_bul_Cyrl,belebele_ces_Latn,belebele_deu_Latn,belebele_ell_Grek,belebele_eng_Latn,belebele_est_Latn,belebele_fin_Latn,belebele_fra_Latn,belebele_heb_Hebr,belebele_hin_Deva,belebele_hun_Latn,belebele_ind_Latn,belebele_ita_Latn,belebele_jpn_Jpan,belebele_kat_Geor,belebele_khm_Khmr,belebele_kor_Hang,belebele_mar_Deva,belebele_mya_Mymr,belebele_nld_Latn,belebele_pes_Arab,belebele_pol_Latn,belebele_por_Latn,belebele_ron_Latn,belebele_rus_Cyrl,belebele_spa_Latn,belebele_swe_Latn,belebele_swh_Latn,belebele_tam_Taml,belebele_tel_Telu,belebele_tha_Thai,belebele_tur_Latn,belebele_ukr_Cyrl,belebele_urd_Arab,belebele_vie_Latn,belebele_yor_Latn,belebele_zho_Hans,belebele_zsm_Latn" --num_fewshot 0 --batch_size auto \
  --output_path "results/belebele_new/$TAG"
lm_eval --model hf --model_args "pretrained=$HF,dtype=bfloat16,trust_remote_code=True" \
  --tasks "include_base_44_arabic,include_base_44_bengali,include_base_44_bulgarian,include_base_44_chinese,include_base_44_dutch,include_base_44_estonian,include_base_44_finnish,include_base_44_french,include_base_44_georgian,include_base_44_german,include_base_44_greek,include_base_44_hebrew,include_base_44_hindi,include_base_44_hungarian,include_base_44_indonesian,include_base_44_italian,include_base_44_japanese,include_base_44_korean,include_base_44_malay,include_base_44_persian,include_base_44_polish,include_base_44_portuguese,include_base_44_russian,include_base_44_spanish,include_base_44_tamil,include_base_44_telugu,include_base_44_turkish,include_base_44_ukrainian,include_base_44_urdu,include_base_44_vietnamese" --num_fewshot 0 --batch_size auto \
  --output_path "results/include_native/$TAG"
