# Aim 1 R1 Technical Repair Addendum

Frozen: 2026-08-26, before any R1 scientific outcome was opened  
Parent protocol: `AIM1_R1_CONTENT_TRANSFER_PROTOCOL_2026-08-24.md`  
Scope: technical completion repair only; the scientific hypotheses and analyses are unchanged

## 1. Trigger

The sealed smoke job `1761877` passed. In the first full array, job `1761878`, nine models completed and ten terminated before scoring because at least one fully tokenized context-target sequence exceeded the 512-token validation ceiling. The observed maximum lengths in the technical error logs were 515 to 768 tokens. No failed task created an output directory. The nine completed outputs remain sealed, and no summary or per-example scientific outcome has been opened.

The failed array indices are `7-12,15-18`: mGPT, OLMo-2-1124-13B, granite-3.1-8b-base, Yi-1.5-9B, TowerBase-7B, occiglot-7b-eu5, Llama-3.1-8B, SmolLM2-360M, Falcon3-3B-Base, and Mistral-Nemo-Base-2407.

## 2. Frozen repair

1. Increase the validation ceiling from 512 to 1024 total tokens.
2. Do not truncate any context or target. The same frozen text is scored in full.
3. Rerun only indices `7-12,15-18`. Do not recompute or overwrite the nine completed model outputs.
4. Keep the model revisions, 100 distinct-document pairs, mismatched-context derangement, 41 languages, BF16 weights, fp32 target-NLL accumulation, batch sizes, and all scientific definitions unchanged.
5. Record `max_sequence_tokens=1024` in repaired manifests. For the nine earlier manifests, absence of this field denotes the original 512-token validation ceiling; their actual sequences already fit and are unchanged by this repair.

The ceiling is an input-validation guard, not a truncation or scoring parameter. Therefore the repair does not alter any successfully scored sequence or outcome.

## 3. Completion and reporting gate

The final analysis must verify both scheduler records: the original 19-task array may fail only at the ten declared repair indices, and every task in the repair array must complete. It must then verify exactly 19 sealed model manifests and their hashes before reading scientific outcomes.

The scheduler query will retry briefly after dependency release because Slurm accounting can lag array completion. No partial outcome assembly is permitted. If any repair task fails for a new reason, stop and document it before another action.

## 4. Claim contract

All original power thresholds, ratio-validity rules, model/family generalization gates, uncertainty analyses, comparator labels, and interpretation limits remain in force. This addendum authorizes no model replacement, hypothesis change, tuning against outcomes, Aim 2 work, or Aim 3 work.
