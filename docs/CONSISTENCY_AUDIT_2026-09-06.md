# Consistency audit, 2026-09-06

Recomputed from saved artifacts on the Mac (JSON/CSV reads only). Compared against the generated paper tables and the ledger values.

## 1. Primary 17-model NTREX grid vs Appendix Table (depth_profiles.tex)
  17 table rows checked, 0 mismatches

## 2. FLORES-200 replication vs flores_replication.tex and text claims
  movers: [('Mistral-7B-v0.3', 2, 15), ('Qwen3-0.6B-Base', 8, 13), ('salamandra-2b', 6, 14)]

## 3. Instruct pairs vs instruct_pairs.tex

## 4. Sentence bootstrap vs dip_bootstrap.tex
  E-A predictions in summary.json: {"P-A1 parity < 0.02 all models": "PASS (14/14)", "P-A2 half-width < 0.03 all models": "FAIL (12/14)", "P-A3 ordering rho>=0.90 in >=95% replicates": "PASS (1.0)"}

## 5. Content transfer (R1) vs paper Table 2
  R1 manifest keys: run_id, created_utc, host, original_array_job_id, repair_array_job_id, final_repair_array_job_id, array_states, n_part_manifests, n_per_example_rows, expected_per_example_rows, n_boot, model_lock_sha256

## 6. Same-support and 33-model exam target
  expanded panel line: EXPANDED PANEL: 2001 rows, 33 models, 73 languages

## 7. Hub test and headline audit

## 8. Deflation refresh (17 models, 1,173 rows)
  hardening.json drop_floor alpha_corr (six exam-capable dev models, n=414): AaR 0.298, MEXA 0.264
  NOTE: no stored artifact reproduces the exam-capable 0.59 (ten models, n=690) quoted in the July report; DECISIONS.md 2026-08 re-derivation gives about 0.49. Treated as unresolved.

## 9. Aim 2 word-alignment collapse arm (WA-C), trained group, layer 8, mean of 3 seeds
  NOTE: the norm figures 451 -> 42 come from the training logs of the arm files and were not re-derived here.

## 10. Evaluation-size dependence (results/estimator_nulls)

## 11. Whitened-basis rank agreement (results/whitened)
  14 models; native LFS keys are dip-layer values; Spearman(native, white) = 0.451

## 12. Misalignment budget (results/budget)
  variant 'plain': 13 models; offset+scale share 0.629 to 0.969; rotation share -0.20% to +0.74%
  variant '_fixed_raw': 14 models; offset+scale share 0.629 to 0.972; rotation share -0.20% to +0.68%
  variant '_selected_centered': 14 models; offset+scale share 0.628 to 0.971; rotation share -1.32% to +0.72%

## 13. Language probe and pooling sensitivity
  max |dip-depth change vs mean pooling| per scheme over 5 models: final 0.143, unitnorm 0.132, lenmatch 0.199

## 14. Strings in the compiled PDFs
| string | ICLR draft | Technical report |
|---|---|---|
| 17 of 17 | yes | yes |
| 19 of 19 | yes | yes |
| 0.039 | yes | yes |
| 0.336 | yes | yes |
| Llama 0.335 | yes | yes |
| null 0.298 | yes | yes |
| FLORES Spearman 0.917/0.92 | yes | yes |
| 14 of 17 | yes | yes |
| median 0.0105 | yes | yes |
| R1 0.508/0.51 | yes | yes |
| R1 CI [0.19, 0.71] | yes | yes |
| model-level 0.770 | yes | yes |
| same-support 0.158 | yes | yes |
| exam q_L 0.041 | yes | yes |
| hub 113 to 124 | yes | yes |
| collapse 451 -> 42 | yes | yes |
| concept var 13,791 | yes | yes |
| LFS 0.4446 | no | yes |
| whitening 0.45 | yes | yes |
| probe 0.875 | yes | yes |

Stale or contest strings (should be absent or explained):
| string | ICLR draft | Technical report |
|---|---|---|
| 0.374 (old Llama depth) | 0 | 1 |
| 0.59 exam-capable | 4 | 1 |
| 124/127 as a general count | 0 | 1 |
| U-shaped | 0 | 2 |
| external baseline | 0 | 0 |
| beats MEXA | 0 | 0 |
| indistinguishable (contest) | 0 | 1 |
| superiority | 0 | 0 |

## Scorecard
| check | artifact value | document value | status |
|---|---|---|---|
| Primary-cohort table rows equal results/grid | 17 models | 0 mismatches | OK |
| Interior minimum + endpoint recovery (primary) | 17/17 | 17/17 | OK |
| Dip depth range | 0.039 (bloom-1b7) to 0.336 (Qwen3-8B-Base); Llama 0.335 | 0.039 bloom-1b7 to 0.336 Qwen3-8B; Llama 0.335 | OK |
| Mistral minimum layer | 2 | 2 | OK |
| Qwen3 depths | 0.206 / 0.213 / 0.314 / 0.336 | 0.206/0.213/0.314/0.336 | OK |
| Dip-layer LFS range | 0.602 to 0.904 | 0.602 to 0.904 | OK |
| Pure-noise null (128,300) | 0.2981 | 0.298 | OK |
| FLORES table rows equal results/flores_grid | 17 rows | 0 mismatches | OK |
| FLORES grid languages (incl. English) | {116} | 116 (English plus 115 non-English) | OK |
| FLORES recovery | 17/17 | 17/17 | OK |
| Depth Spearman NTREX vs FLORES | 0.917 | 0.917 | OK |
| Min layer within 2 | 14/17 | 14/17 | OK |
| Median |depth diff| | 0.0105 | 0.0105 | OK |
| Profile r range | 0.936 to 1.000 | 0.936 to 1.000 | OK |
| granite depth NTREX -> FLORES | 0.190 -> 0.074 | 0.190 -> 0.074 | OK |
| Instruct table rows equal artifacts | 6 rows | 0 mismatches | OK |
| Instruct recovery | 6/6 | 6/6 | OK |
| Max |depth diff| base vs instruct | 0.0078 | <= 0.008 | OK |
| Final-layer LFS higher after tuning | 5/6 | 5/6 | OK |
| Min layer within 2 (instruct) | 5/6 | 5/6 | OK |
| Bootstrap table rows equal artifacts | 14 rows | 0 mismatches | OK |
| Half-widths 0.004-0.020 | 12/14 models; others: [('Mistral-7B-v0.3', 0.045), ('salamandra-2b', 0.059)] | 12 of 14; Mistral 0.045, salamandra-2b 0.059 | OK |
| Parity dump vs grid | max abs diff 7.8e-05 | < 1e-4 | OK |
| q_L vs R_content, model bootstrap | 0.508 [0.188, 0.707] | 0.508 [0.188, 0.707] | OK |
| q_L crossed / family intervals | [0.181, 0.723] / [0.242, 0.757] | [0.181, 0.723] / [0.242, 0.757] | OK |
| MEXA reference, model bootstrap | 0.591 [0.330, 0.818] | 0.591 [0.330, 0.818] | OK |
| AaR tail reading, model bootstrap | 0.443 [0.093, 0.709] | 0.443 [0.093, 0.709] | OK |
| q_L vs raw content, model bootstrap | 0.620 [0.337, 0.769] | 0.620 [0.337, 0.769] | OK |
| Model-level q_L vs median R_content | 0.770 [0.440, 0.941] | 0.770 [0.440, 0.941] | OK |
| Leave-one-family-out range (q_L) | 0.439 to 0.598, 13 deletions, all positive=True | 0.439 to 0.598, all positive | OK |
| Same-support alpha, LFS concept direction (language / model CI) | 0.158 [0.034, 0.285] / [-0.020, 0.289] | 0.158 [0.034, 0.285] / [-0.020, 0.289] | OK |
| Legacy 4-model content transfer, LFS concept direction | 0.842 [0.790, 0.905] | 0.842 [0.790, 0.905] | OK |
| 33-model pooled exam: q_L | 0.041 [-0.045, 0.121] | 0.041 [-0.045, 0.121] | OK |
| 33-model pooled exam: MEXA / AaR | 0.238 / 0.105 | 0.238 / 0.105 | OK |
| Latent factor beats English, per model | 113 to 124 of {127} over 17 models; lowest ['bloom-1b7'] | 113 to 124 of 127 | OK |
| Headline audit: direct-SS interior+recovery | 17/17 | 17/17 | OK |
| Headline audit: REML LFS-VC interior+recovery | 19/19 | 19/19 | OK |
| AaR@10 raw / alpha; n | 0.540 / 0.208; n=1173 | 0.540 / 0.208; 1173 | OK |
| MEXA raw / alpha | 0.539 / 0.176 | 0.539 / 0.176 | OK |
| Exam-capable 0.59 traceable to a stored artifact | not found locally | 0.59 (report text) | MISMATCH |
| WA-A LFS / MEXA / raw V_concept | 0.4446 / 0.4444 / 13791 | 0.4446 / 0.4444 / 13791 | OK |
| WA-C LFS / MEXA / raw V_concept | 0.4932 / 0.4878 / 699 | 0.4932 / 0.4878 / 699 | OK |
| WA-G LFS / MEXA / raw V_concept | 0.4826 / 0.3944 / 13792 | 0.4826 / 0.3944 / 13792 | OK |
| LFS drift N=100->1000 (4 models) | -0.010 to -0.006 | -0.006 to -0.010 | OK |
| MEXA drift N=100->1000 (4 models) | -0.125 to -0.065 | -0.065 to -0.125 | OK |
| Whitened vs native rank agreement | 0.45 over 14 models | 0.45 | OK |
| Offset+scale share range (plain variant, language mean per model) | 0.63 to 0.97 | 0.65 to 0.98 | OK |
| Rotation share range (plain variant) | -0.20% to +0.74% | -0.2% to +0.7% | OK |
| Linear language-probe accuracy across layers (Qwen3-0.6B) | 0.875 to 0.886 | 0.875 to 0.886 | OK |
| Pooling: max dip-depth change vs mean pooling, across schemes | 0.13 to 0.20 | 0.13 to 0.20 | OK |

49 of 50 checks OK.
