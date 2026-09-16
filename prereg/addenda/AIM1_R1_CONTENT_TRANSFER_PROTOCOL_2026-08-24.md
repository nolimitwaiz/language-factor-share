# Aim 1 R1 Addendum: A9 Content-Transfer Expansion

Frozen: 2026-08-24, before any R1 model outcome is computed or opened  
Scope: Aim 1 section 3.1.3 only  
Status at freeze: R0 complete; R1 model runs not started

## 1. Question and claim contract

The legacy section 3.1.3 result found a large association between the LFS concept-direction reading and held-out cross-lingual content transfer, but it contained only four independent models. R1 asks whether that relationship generalizes to the 19 successful A9 models.

The result can support a model-breadth claim about this fixed content-transfer harness. It cannot establish universal downstream validity, instruction-following quality, retrieval quality, or Aim 2/Aim 3 performance.

## 2. Frozen models

The exact 19 tags, Hugging Face IDs, cached commit revisions, and model-family labels are frozen in `manifests/R1_A9_MODEL_REVISION_LOCK_2026-08-24.tsv`. No failed model will be replaced after outcomes.

There are 13 model families. The four Qwen2.5 sizes, three XGLM sizes, and two Sailor sizes are not treated as nine independent families in family-sensitivity analysis.

## 3. Frozen dataset and pair construction

Use NTREX `newstest2019` with:

- English target sentence `s2`;
- immediately preceding matched context `s1` from the same document;
- the translation of `s1` in each of 41 frozen context languages, including English;
- an explicitly different-document mismatched context in the same language.

The 41-language order is the committed `transfer_313b.py` order. All 41 files must exist and have 1,997 aligned rows.

Pair selection is repaired relative to the legacy script:

1. An adjacent pair is eligible when `s1` and `s2` share a document ID, `s1` has at least five whitespace-delimited English words, and `s2` has 8–45 words.
2. Keep only the first eligible pair from each document in corpus order.
3. Freeze the first 100 distinct documents. The unit is therefore one pair per document, rather than 100 pairs dominated by a few documents.
4. Assign the 100 mismatched contexts with a deterministic minimum-cost derangement. The cost prioritizes absolute English-context word-length difference, then index distance, then candidate index. Self/document matches receive prohibitive cost.
5. Each mismatched context is used exactly once, and its document must differ from the target pair's document.

The builder writes immutable pair and dataset manifests before any model run. Model jobs must verify those hashes.

The data-lock job `1761872` completed before any model outcome. The frozen pair file is `results/round3/r1_content_pairs_2026-08-24/pairs.csv` with SHA-256 `986d4f3a8730b7e8087ebf0d558de72e8735aba3b4fb5493a3f1e04b42bc1112`. Its manifest hash is `7c62075cdda2cecd4dfb3c3e06664c6fc9eae17e487eb4ef9c01be5a41d05fcd`. It contains 100 unique matched and 100 unique mismatched documents. Mean absolute English context-length mismatch is 0.34 words; the maximum is 12 words.

The final model-revision-and-batch lock SHA-256 is `dcb3c1fbaffa8cca70a4090b3b62d269a5daddc43779e62a9e2cdede8679930c`. The earlier revision-only hash was superseded before any model outcome was run.

## 4. Frozen scoring

For every model, pair, and language, compute mean target-token negative log likelihood for:

- no context;
- matched same-document context;
- mismatched different-document context.

The target is always English. Context and target are separated by one newline. Scoring is deterministic teacher forcing; there is no sampling or free generation. NLL is accumulated in fp32 over target tokens only. All per-example values, target-token counts, context-token counts, failures, model revision, and input hashes are retained.

For language `L`:

- `delta_matched_L = mean(NLL_none - NLL_matched_L)`;
- `delta_mismatched_L = mean(NLL_none - NLL_mismatched_L)`;
- `content_L = mean(NLL_mismatched_L - NLL_matched_L)`;
- `R_content_L = content_L / content_eng` only when `content_eng > 0`.

The raw `content_L` is always reported. Ratios are never silently clipped or sign-flipped.

## 5. Primary and secondary analysis

Primary candidate: the existing A9 `q_L = 1 - LFS-VC` reading. No RMFS scalar is tested in R1.

Primary target: `R_content_L` on non-English common-support rows. Residualize candidate and target using the existing frozen attribution controls: language exposure, macro-family, script, and tokenizer fertility. Report deflated Spearman correlation with a model-cluster bootstrap as primary and a crossed model-language bootstrap as secondary.

Required robustness analyses:

1. model-level Spearman between `q_L` and each model's median valid `R_content_L`;
2. the same analysis using raw `content_L` instead of the ratio;
3. leave-one-model-family-out estimates;
4. model-family bootstrap as a conservative sensitivity analysis;
5. the same-support MEXA and AaR external comparators, labeled external rather than original metrics;
6. paired document bootstrap within model-language for the uncertainty of each content estimate.

The 19-model A9 panel is primary. A combined 23-model table that appends the four historical models is secondary because those outcomes were already opened and were generated by the legacy pair selection.

## 6. Power and interpretation

With 19 independent models, a Fisher-transform approximation gives about 80% power for a correlation near `0.60` under two-sided alpha 0.05. With 13 families, the analogous detectable effect is roughly `0.68`. R1 is therefore capable of checking whether the earlier large effect generalizes, but it is not well powered for modest correlations.

Do not interpret a wide interval as proof of no relationship. Do not interpret a positive language-cluster interval as model generalization if the model- or family-level interval crosses zero.

## 7. Completeness, failures, and stop rules

- Run a technical smoke on SmolLM2-360M using two pairs and three languages. The smoke is excluded from analysis and must not print outcome values.
- Launch the 19-model array only after the smoke passes finite-value, schema, revision, and hash checks.
- A model is ratio-valid only if `content_eng` is finite and positive and at least 32 of 40 non-English languages have valid estimates.
- Technical failures remain missing and are not replaced.
- If fewer than 15 models or 10 model families are ratio-valid, no model-generalization claim is made; raw-content results and failures are still reported.
- All 19 model tasks must terminate before primary outcomes are assembled or inspected.
- No Aim 2 or Aim 3 run may be attached to this array.

## 8. Compute estimate

The full design scores 100 no-context sequences and 8,200 context-conditioned sequences per model: 8,300 sequences per model and 157,700 across 19 models, approximately 7–10 million input tokens depending on tokenizer fertility. Expected support is one A100-class GPU per model task, with dynamic batch sizes by model width. The smoke records throughput before the full time limit is finalized.

## 9. Reporting gate

Before any paper edit:

- verify 19/19 task manifests or document every failure;
- freeze output hashes and completeness JSON;
- run the predeclared model, crossed, family, and document uncertainty analyses;
- compare the expanded result with the legacy four-model estimate without pooling their pair designs as if identical;
- report the result regardless of direction;
- obtain Professor Koehn's agreement on the final content-transfer claim.

## 10. Execution record

- Pair-lock CPU job: `1761872`, complete.
- Sealed SmolLM2-360M smoke: `1761877`, complete.
- Frozen 19-model array: `1761878`. Nine tasks completed; ten encountered the documented technical sequence-length ceiling.
- 1,024-token technical repair array: `1765008`. Eight tasks completed; Yi-1.5-9B and Mistral-Nemo remained above the engineering ceiling. See `AIM1_R1_TECHNICAL_REPAIR_2026-08-26.md`.
- 2,048-token final repair array: `1765050`, both remaining tasks complete. See `AIM1_R1_FINAL_CEILING_REPAIR_2026-08-26.md`.
- Authoritative completion-gated analysis: `1765948`, complete. It validated all three array histories, 19 part manifests, 157,700 per-example rows, frozen input hashes, and eight output hashes before results were inspected.
- Result record: `docs/AIM1_R1_CONTENT_TRANSFER_RESULTS_2026-08-26.md`.
