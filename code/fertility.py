#!/usr/bin/env python3
"""Tokenizer fertility per (model, language) on NTREX text.

fertility(model, lang) = (tokens/char in lang) / (tokens/char in eng),
computed on the same 100 parallel sentences. Keyed to FLORES codes via the
Belebele metadata + ISO mapping.
"""
import csv, os
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTREX = os.path.join(ROOT, 'data', 'NTREX', 'NTREX-128')

MODELS = ['Qwen/Qwen3-0.6B-Base', 'allenai/OLMo-2-0425-1B',
          'mistralai/Mistral-7B-v0.3', 'bigscience/bloom-1b7',
          'utter-project/EuroLLM-1.7B']
# family members share tokenizers: Qwen3-* -> Qwen3-0.6B; OLMo-2-* -> OLMo-2-1B;
# bloom-* -> bloom-1b7. Recorded in SHARED below and expanded at output time.
SHARED = {
    'Qwen/Qwen3-0.6B-Base': ['Qwen3-0.6B-Base', 'Qwen3-1.7B-Base',
                             'Qwen3-4B-Base', 'Qwen3-8B-Base'],
    'allenai/OLMo-2-0425-1B': ['OLMo-2-0425-1B', 'OLMo-2-1124-7B'],
    'mistralai/Mistral-7B-v0.3': ['Mistral-7B-v0.3'],
    'bigscience/bloom-1b7': ['bloom-1b7', 'bloom-7b1'],
    'utter-project/EuroLLM-1.7B': ['EuroLLM-1.7B'],
    # held-out families (walk-forward) + A3 additions + campaign holdout
    'HuggingFaceTB/SmolLM2-1.7B': ['SmolLM2-1.7B'],
    'tiiuae/Falcon3-7B-Base': ['Falcon3-7B-Base'],
    'BSC-LT/salamandra-2b': ['salamandra-2b', 'salamandra-7b'],
    'ibm-granite/granite-3.1-8b-base': ['granite-3.1-8b-base'],
    '01-ai/Yi-1.5-9B': ['Yi-1.5-9B'],
    'meta-llama/Llama-3.1-8B': ['Llama-3.1-8B'],
}

# FLORES code -> NTREX filename code (iso639-3 with occasional suffix)
def flores_to_ntrex(code):
    iso = code.split('_')[0]
    special = {'zho': 'zho-CN', 'aze': 'aze-Latn', 'srp': 'srp-Cyrl',
               'zsm': 'msa', 'pes': 'fas', 'npi': 'nep', 'ory': 'ori',
               'lvs': 'lav', 'als': 'sqi', 'azj': 'aze-Latn', 'khk': 'mon',
               'pbt': 'pus', 'swh': 'swa', 'plt': 'mlg', 'kmr': 'kur',
               'ydd': 'yid', 'uzn': 'uzb', 'gaz': 'orm', 'ckb': 'ckb'}
    return special.get(iso, iso)


def load_langs():
    metap = os.path.join(ROOT, 'results', 'deflation', 'language_meta.csv')
    with open(metap) as f:
        return [r['flores_code'] for r in csv.DictReader(f)]


def main():
    langs = load_langs()
    texts = {}
    for fc in langs + ['eng_Latn']:
        nt = 'eng' if fc == 'eng_Latn' else flores_to_ntrex(fc)
        for cand in (f'newstest2019-ref.{nt}.txt', f'newstest2019-src.{nt}.txt'):
            p = os.path.join(NTREX, cand)
            if os.path.exists(p):
                with open(p) as f:
                    texts[fc] = ' '.join(l.strip() for l in list(f)[:100])
                break
    print(f'[text] {len(texts)-1}/{len(langs)} Belebele languages mapped to NTREX')

    rows = []
    for hf_name, members in SHARED.items():
        tok = AutoTokenizer.from_pretrained(hf_name)
        eng_rate = len(tok.encode(texts['eng_Latn'])) / len(texts['eng_Latn'])
        for fc, txt in texts.items():
            rate = len(tok.encode(txt)) / len(txt)
            fert = rate / eng_rate
            for m in members:
                rows.append([fc, m, round(fert, 4)])
        print(f'[fert] {hf_name}: done ({len(texts)} langs)')

    out = os.path.join(ROOT, 'results', 'deflation', 'fertility.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['flores_code', 'model', 'fertility'])
        w.writerows(rows)
    print(f'[done] -> {out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
