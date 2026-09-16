# Aim 1 R1 Final Token-Ceiling Repair Addendum

Frozen: 2026-08-26, before the final retry and before any R1 scientific outcome was opened  
Parents: `AIM1_R1_CONTENT_TRANSFER_PROTOCOL_2026-08-24.md` and `AIM1_R1_TECHNICAL_REPAIR_2026-08-26.md`  
Scope: final tokenizer-length guard repair for Yi-1.5-9B and Mistral-Nemo-Base-2407 only

## 1. Trigger and preflights

The 1024-token repair array `1765008` completed eight of ten tasks. Yi-1.5-9B and Mistral-Nemo-Base-2407 stopped before scoring on 1,130- and 1,108-token sequences respectively. Neither created an output directory.

Before changing the guard again, tokenizer-only preflights evaluated all 8,300 frozen inputs for each model using the locked revisions and input hashes.

| Model | Above 512 | Above 1024 | Above 2048 | Maximum | Maximum cell |
|---|---:|---:|---:|---:|---|
| Yi-1.5-9B | 61 | 2 | 0 | 1,142 | pair 58, Khmer, matched |
| Mistral-Nemo-Base-2407 | 13 | 2 | 0 | 1,120 | pair 58, Khmer, matched |

The generated technical records are `cluster_logs/aim1_r1_yi_length_preflight.json` and `cluster_logs/aim1_r1_mistral_length_preflight.json`. They contain lengths and cell identifiers only, not NLLs or scientific outcomes.

## 2. Frozen final repair

1. Set the validation ceiling to 2048 total tokens.
2. Rerun only array indices 10 and 18.
3. Do not truncate text and do not alter model revisions, data, pairs, precision, batch sizes, target-token scoring, or any scientific definition.
4. Preserve all 17 completed model outputs. The final retries may write outputs only because their prior attempts produced none.
5. Keep the current tokenizer construction unchanged. In particular, do not act on the Transformers Mistral-regex warning inside this repair; changing tokenizer behavior would be a different preprocessing decision and requires separate analysis.

The ceiling remains a fail-fast validation guard rather than a scoring transformation. The full frozen text is identical across attempts.

## 3. Final gate

Analysis must verify the original array, the 1024 repair array with failures permitted only at indices 10 and 18, the completed two-task 2048 repair, and exactly 19 sealed model manifests before reading any scientific outcome. No further ceiling change is authorized without another pre-outcome technical record.
