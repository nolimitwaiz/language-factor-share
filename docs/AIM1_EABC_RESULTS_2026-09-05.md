# Aim 1 E-A / E-B / E-C results: dip-depth bootstrap, FLORES-200 replication, instruct versus base

**Date:** 2026-09-05
**Protocol (frozen before submission):** `prereg/addenda/AIM1_EABC_PROTOCOL_2026-09-05.md`
**Jobs (CLSP, all COMPLETED, exit 0:0):** E-A array 1784646 (14 cpu tasks, 0:46 to 4:15 each);
E-B smoke 1784647 (6:31) and array 1784648 (17 A100 tasks, 7:40 to 51:47 each);
E-C array 1784649 (7 A100 tasks, 13:42 to 28:39 each). About 8 GPU-hours total.
**Scorer:** `src/analysis/score_eabc.py` on CLSP; outputs `results/eabc_scoring/summary.{json,md}`
(SHA-256 `fcdf3d6f…7a3d14` and `1f5dc0e5…ab2e96`).
**Raw artifacts:** `results/dip_bootstrap/<model>/{bootstrap.json,manifest.json}` (14 models),
`results/flores_grid/<model>/{metrics.json,manifest.json}` (17), `results/instruct_grid/<model>/…` (7),
`data/FLORES200/ntrex_to_flores.json` (mapping). Every manifest records the model snapshot hash,
input hashes, dtype, GPU, and wall time. No existing artifact was modified.

## Scorecard against the frozen predictions

| Prediction | Result | Verdict |
|---|---|---|
| P-A1 parity: dump first-300 LFS within 0.02 of grid at layer 0 and dip | max difference 7.8e-5 (14/14) | PASS |
| P-A2 bootstrap half-width of dip depth < 0.03 for every model | 12/14; Mistral-7B 0.045, Salamandra-2B 0.059 | **FAIL** |
| P-A3 ordering Spearman ≥ 0.90 in ≥ 95% of replicates | 100% of 2,000 replicates | PASS |
| P-B1 ≥ 110 NTREX languages map to FLORES-200 | 115 of 127 | PASS |
| P-B2 endpoint recovery on FLORES in 17/17 | 17/17 | PASS |
| P-B3 Spearman(dip depth NTREX, FLORES) ≥ 0.80 | 0.917 | PASS |
| P-B4 minimum layer within 2 for ≥ 14/17 | 14/17 | PASS |
| P-B5 median absolute depth difference < 0.05 | 0.0105 | PASS |
| P-C1 endpoint recovery for all six instruct models | 6/6 | PASS |
| P-C2 final-layer LFS higher in instruct for ≥ 4/6 | 5/6 | PASS |
| P-C3 minimum layer within 2 for ≥ 4/6 | 5/6 | PASS |
| P-C4 absolute dip-depth change < 0.05 for ≥ 4/6 | 6/6 (max 0.008) | PASS |

Eleven of twelve frozen predictions passed. The one failure is reported as scored and is
informative (below).

## E-A: what the bootstrap says

Per model (dip depth median over 2,000 subsamples of 300 sentences, 95 percent interval):

| model | dip layer | grid depth (first 300) | median | interval | half-width |
|---|---:|---:|---:|---|---:|
| Qwen3-8B-Base | 19 | 0.336 | 0.352 | [0.341, 0.363] | 0.011 |
| Qwen3-4B-Base | 17 | 0.314 | 0.338 | [0.324, 0.351] | 0.014 |
| salamandra-2b | 6 | 0.254 | 0.313 | [0.254, 0.372] | 0.059 |
| Mistral-7B-v0.3 | 2 | 0.212 | 0.246 | [0.199, 0.289] | 0.045 |
| Qwen3-1.7B-Base | 13 | 0.213 | 0.229 | [0.219, 0.239] | 0.010 |
| Qwen3-0.6B-Base | 8 | 0.206 | 0.225 | [0.205, 0.246] | 0.020 |
| salamandra-7b | 18 | 0.180 | 0.187 | [0.181, 0.195] | 0.007 |
| Falcon3-7B-Base | 13 | 0.140 | 0.153 | [0.141, 0.167] | 0.013 |
| OLMo-2-1124-7B | 12 | 0.135 | 0.140 | [0.132, 0.149] | 0.009 |
| EuroLLM-1.7B | 14 | 0.097 | 0.101 | [0.096, 0.107] | 0.005 |
| OLMo-2-0425-1B | 9 | 0.077 | 0.081 | [0.073, 0.090] | 0.008 |
| SmolLM2-1.7B | 15 | 0.068 | 0.075 | [0.067, 0.083] | 0.008 |
| bloom-7b1 | 17 | 0.043 | 0.048 | [0.044, 0.053] | 0.004 |
| bloom-1b7 | 14 | 0.039 | 0.043 | [0.039, 0.049] | 0.005 |

Three readings.

1. **The headline numbers are the data.** The first 300 sentences of each dump reproduce the
   grid's layer-0 and dip-layer LFS to better than 1e-4 in every model, so the bootstrap speaks
   directly to the published values.
2. **Precision is high except where the minimum is very early.** Twelve models have intervals
   narrower than ±0.02. Mistral-7B (minimum at layer 2 of 32) and Salamandra-2B (layer 6 of 24)
   have intervals of ±0.045 and ±0.059: an early, sharp minimum depends more on which sentences
   are sampled than a mid-network one does. This is a real property of those two profiles, not
   noise in the estimator, and the frozen bar failed for exactly those two models.
3. **Random subsets give slightly deeper dips than the first 300 sentences,** by 0.004 to 0.06
   (median about 0.01), most for Salamandra-2B (0.254 versus 0.313). The first 300 sentences come
   from the beginning of a document-ordered corpus and span fewer documents; random subsets span
   more concept diversity, which raises the concept sum of squares at the dip. The published
   depths are therefore conservative; the cross-model ordering is unaffected (Spearman ≥ 0.90 in
   all 2,000 replicates against the published ordering).

## E-B: FLORES-200 replication

All 17 models recover at both endpoints on FLORES-200; dip depths agree with NTREX at Spearman
0.917 with a median absolute difference of 0.011; per-layer profiles correlate at Pearson 0.936 to
1.000. Details in `results/eabc_scoring/summary.md` and the paper's Appendix.

Where the two corpora differ, and why it matters:

- **Minimum layer.** Three models moved their argmin by more than two layers: Qwen3-0.6B (8 to
  13), Mistral-7B (2 to 15), Salamandra-2B (6 to 14). All three have a plateau or an early sharp
  dip whose neighboring layers are within a few thousandths of the minimum, so the argmin is
  fragile even when the profile is not (profile correlations 0.97, 0.94, 0.94). The paper should
  describe minimum location for these models as "early or plateau" rather than a single layer.
- **Depth.** Granite is the outlier: 0.190 on NTREX, 0.074 on FLORES (profile r 0.946, largest
  per-layer difference 0.116). Mistral-7B (0.212 to 0.180) and Salamandra-2B (0.254 to 0.176) also
  shallow; Qwen3-8B and Llama-3.1-8B deepen (0.336 to 0.372, 0.335 to 0.372). Granite's dip depth
  is corpus-sensitive and its rank should be stated with that caveat.
- **Ordering.** Rank agreement 0.92 across 17 models; the deep (Qwen3-4B/8B, Llama) and shallow
  (BLOOM, SmolLM2, OLMo-2-1B, EuroLLM) ends are identical on both corpora.

This converts "one corpus" from a limitation into a replication, and FLORES is the corpus MEXA
reports on, so it also removes a comparability objection.

## E-C: instruction tuning

| pair | depth base | depth instruct | min layer base → instruct | final LFS base → instruct |
|---|---:|---:|---|---|
| Qwen3-0.6B | 0.206 | 0.211 | 8 → 8 | 0.958 → 0.956 |
| Qwen3-1.7B | 0.213 | 0.216 | 13 → 8 | 0.960 → 0.963 |
| Qwen3-4B | 0.314 | 0.320 | 17 → 17 | 0.956 → 0.957 |
| Qwen3-8B | 0.336 | 0.343 | 19 → 18 | 0.958 → 0.960 |
| Qwen2.5-7B | 0.287 | 0.295 | 4 → 4 | 0.959 → 0.961 |
| Llama-3.1-8B | 0.335 | 0.332 | 14 → 14 | 0.936 → 0.937 |

Instruction tuning leaves the LFS profile essentially unchanged: dip depth moves by at most
0.008, the final-layer share by at most 0.003 (up in five of six, as predicted, but by amounts
that are within the sentence-bootstrap interval), and the minimum layer is identical or adjacent
except for the Qwen3-1.7B plateau (13 to 8, same argmin fragility as above). The profile is a
property of pretraining. This answers the "only base models" objection and is a finding worth
one paragraph: post-training changes behavior a great deal and the language-versus-meaning
geometry of the residual stream very little.

Note: Qwen2.5-7B base was measured here on the 128-language NTREX grid for the first time (the
expansion cohort used a 41-language protocol); its dip is at layer 4 of 28 with depth 0.287,
another early-minimum model.

## What changed in the paper and report

- Paper: the held-out-families paragraph now carries the FLORES replication and the bootstrap
  intervals; a new "Instruction tuning" paragraph in the stress-test section; a new appendix with
  the three tables; abstract mentions the replication. Main text still ends on page 9.
- Technical report: new subsection in Results I with the same content.
- Claims ledger rows C52 to C54 updated with values and job IDs.

## Claim boundary

These results support: the profile shape and the cross-model ordering of dip depth are stable to
the choice of parallel corpus and to the sampled sentences, and are unchanged by instruction
tuning. They do not support: exact minimum-layer claims for plateau or early-dip models, exact
dip-depth values for granite across corpora, or anything about downstream behavior.


**Language count note (2026-09-06).** The FLORES grid has 116 languages: English plus the 115 mapped non-English NTREX languages. Where this document says 115 it counts non-English languages only; the paper and technical report now say 116 including English.
