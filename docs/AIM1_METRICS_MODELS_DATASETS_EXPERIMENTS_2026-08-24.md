# Aim 1 Metric, Dataset, Model, and Experiment Registry

**Date:** 2026-08-24  
**Scope lock:** Aim 1 only. No Aim 2 optimization campaign and no Aim 3 retrieval/RAG campaign may begin before the Aim 1 package is complete.  
**Primary source:** `paper/TECHNICAL_REPORT.tex` and its 39-page rendered PDF.  
**Later corrections:** `docs/LFS_SUBMISSION_REVERIFICATION_2026-08-20.md`, `prereg/addenda/ROUND3_EXECUTION_PROTOCOL_2026-08-24.md`, and `docs/ROUND3_PHASE_A_CORE_RESULTS_2026-08-24.md`.  
**Evidence rule:** when prose conflicts with a frozen protocol, producing code, or raw result, the protocol/code/result wins.

## 1. Current scientific decision

The ICLR paper is LFS-led.

- The primary estimator is direct sums-of-squares LFS.
- REML LFS-VC is a robustness estimator, not a silent replacement.
- AaR, the English-Hub analysis, the mapping ladder, and raw preservation readings answer different Aim 1 questions.
- Frozen RMFS v1 is a historical stress test and negative result. It did not beat LFS on the shared validation frames.
- Profile 0.2 is a four-reading diagnostic profile. It is not a scalar, is not to be averaged, and is not yet outcome-validated as a complete profile.
- MEXA is an external published baseline. It is never described as an original project metric.
- Archived intervention checkpoints may be used as controlled Aim 1 stress-test inputs. Doing so does not make the present work an Aim 2 campaign.

The central Aim 1 question is:

> How much systematic representation variation is associated with language rather than matched meaning, how does that structure change through depth, and which complementary readings reveal failure modes that LFS alone cannot see?

## 2. Status vocabulary

| Status | Meaning |
|---|---|
| **Verified complete** | Producing code and artifacts exist; a later audit reproduced or checked the result. |
| **Complete with correction** | The experiment ran, but later auditing narrowed or corrected the original claim. |
| **Computed, analysis open** | Metric outputs exist, but the controlled statistical interpretation is not frozen. |
| **Required, not run** | Needed for the intended Aim 1 paper claim and still requires execution. |
| **Optional/gated** | Potentially useful, but not authorized until a protocol, power analysis, and compute estimate are accepted. |
| **Deferred by scope** | Not part of the current Aim 1 completion gate. |
| **Retired/negative** | Preserved as evidence, but not used as the new headline metric or claim. |

## 3. Metric and diagnostic registry

### 3.1 Primary Aim 1 metric: direct-SS LFS

Input is a balanced representation grid `z[l,c]` with languages `l = 1...L`, matched concepts or sentences `c = 1...N`, and hidden dimension `D`.

Standard pipeline:

1. Extract the same concept in every language at a fixed model layer.
2. Pool token hidden states in fp32.
3. Jointly standardize every coordinate over all language-concept cells for that layer.
4. Compute the grand mean, language means, and concept means.
5. Compute:
   - `SS_language = N * sum_l ||mean_language[l] - grand_mean||^2`
   - `SS_concept = L * sum_c ||mean_concept[c] - grand_mean||^2`
   - `SS_residual = sum_l,c ||z[l,c] - mean_language[l] - mean_concept[c] + grand_mean||^2`
6. Report `LFS = SS_language / (SS_language + SS_concept)`.
7. Always report the two numerator/denominator components, residual, grid dimensions, preprocessing, pooling, layer, norm, and rank diagnostics beside the ratio.

Interpretation:

- Larger LFS means language main-effect variation is larger relative to language plus concept main-effect variation on that fixed grid.
- Smaller LFS means concept main effects occupy a larger share of the two systematic main effects.
- Smaller LFS does not automatically mean a better, more capable, or language-agnostic model.
- Residual interaction is outside the ratio by design. Rotations, nonlinear language-concept interactions, and unmodeled structure can therefore require separate instruments.
- The balanced Gaussian pure-noise expectation is `(L - 1) / (L + N - 2)`. Absolute LFS values must not be compared across different `L`, `N`, pooling, or preprocessing without calibration.

Status: **verified complete as the paper estimator**.

### 3.2 LFS layer-profile statistics

| Reading | Computation | Use | Constraint |
|---|---|---|---|
| Layerwise LFS curve | Direct LFS at every saved layer | Representation fingerprint | Do not call every curve a formal U-shape without a frozen smoothness rule. |
| Interior minimum | Argmin LFS over measured layers | Location of the most concept-dominant layer under LFS | Historical code also used argmax mean MEXA as a "best layer"; keep the rules distinct. |
| Dip depth | Layer-0 LFS minus minimum LFS | Within-model change through depth | Comparisons require the same estimator and grid. |
| Endpoint recovery | First and last layer both above the interior minimum | Robust descriptive profile claim | Audited separately for direct SS and REML cohorts. |

Current verified wording: 17/17 direct-SS profiles and 19/19 A9 REML profiles have an interior minimum with both endpoints above it. The cohorts and estimators must remain separate.

### 3.3 Raw factor and preservation readings

These are scale-sensitive companions to LFS.

| Reading | Definition/use | Main failure it detects |
|---|---|---|
| Raw `SS_language`, `SS_concept`, `SS_residual` | Same decomposition before or alongside normalization | Tells whether a ratio change came from language, concept, or residual movement. |
| Within-concept spread | Mean squared spread around each language mean | Collapse or reorganization within languages. |
| Shared-concept energy | Mean squared magnitude of concept means around the grand mean | Loss of the shared additive concept signal. |
| Mean vector norm | Mean raw hidden-state norm | Uniform shrinkage. |
| CVP/preservation ratio | Current concept variance divided by a frozen reference value | Absolute concept collapse that a scale-invariant ratio misses. |
| Singular-value entropy rank | Entropy effective rank from singular-value weights | Dimensional collapse; named separately from covariance rank. |
| Covariance-eigenvalue entropy rank | Entropy effective rank from covariance eigenvalues | Dimensional collapse; not interchangeable with singular entropy rank. |
| Participation ratio | Spectrum-based effective dimensionality | Rank concentration and low-rank damage. |

Status: **verified construct diagnostics**. CVP detects reorganization, not guaranteed task harm.

### 3.4 Alignment-at-Risk and margin readings

For one target language and an English pivot:

1. Normalize sentence vectors.
2. Form the target-to-English cosine similarity matrix.
3. For each sentence, calculate `margin = similarity(true pair) - maximum similarity(wrong pair)`.
4. Mean margin summarizes average matching separation.
5. AaR@q averages the worst `q` fraction of margins. The project uses AaR@10 primarily and AaR@{1,5,10,20} in the 1,500-sentence tail campaign.

Uses:

- Mean margin: average pairwise alignment.
- AaR: tail failure among sentences.
- Weak-language tail: average of the lowest quarter of per-language AaR values in Profile 0.2.

Constraints:

- AaR is retrieval-based and depends on candidate-set size.
- AaR and cosine margins are invariant to uniform positive scaling of each complete vector, so they cannot detect pure shrinkage without a raw preservation reading.
- AaR is a project metric. MEXA is not.

Status: **verified tail diagnostic**.

### 3.5 English-Hub analysis

Question: is the shared representation factor specifically English, or a multilingual latent factor?

Process:

1. At the frozen selected layer, reduce representations to a common PCA space.
2. For every target language, cross-validate a ridge map from English representations to that target.
3. Cross-validate a second map from the leave-target-out multilingual mean to the target.
4. Compare held-out R-squared.
5. Rank every individual donor language as a control against the fact that an average is smoother than one language.

Status: **complete with correction**. The original smoothness-confounded advantage was corrected. Latent-over-English counts vary by model from 113/127 to 124/127; any 124/127 statement must name the model.

### 3.6 Mapping-complexity ladder and misalignment budget

Nested cross-language maps are fit on training concepts and evaluated on held-out concepts:

| Rung | Map |
|---|---|
| M0 | Identity/shared space only |
| M1 | Additive offset |
| M2 | Offset plus isotropic scale |
| M3 | Similarity/orthogonal rotation |
| M4 | General linear ridge map |
| M5 | Nonlinear RBF kernel ridge map |

Each rung receives its held-out reduction in reconstruction error relative to M0. Negative held-out shares are retained as overfit signals. Permutation nulls are required. Document-purged splits are mandatory for new claims.

Status: **complete with correction**. Offset plus scale dominate the audited geometry, rotation is negligible, but the old nonlinear percentages were biased by same-document leakage. A document-purged Qwen audit preserves the main additive conclusion while making the nonlinear rung more negative.

### 3.7 Transfer alpha and downstream risk statistics

These are validation targets/statistics, not representation metrics.

- Transfer alpha: residualized downstream performance after controlling for model fixed effects, log CulturaX tokens, macro-family, script, and tokenizer fertility. The four-choice Belebele chance floor is respected through the outcome transform.
- PaR@10: worst-decile downstream accuracy across languages. It measures service to the language tail.
- Content-transfer benefit: target NLL improvement from matched foreign-language context, subtracting the mismatched same-language context effect and normalizing when specified.
- Content attention: matched-minus-mismatched attention-to-context ratio after dividing by the uniform-attention geometric baseline.

Status: **completed validation instruments**, with target-specific and model-cluster limitations.

### 3.8 MEXA and other external comparators

| Comparator | Role | Rule |
|---|---|---|
| MEXA | Published sentence-retrieval baseline | External baseline only; never "our metric." |
| Linear CKA | Normalized similarity comparator in invariance/battery work | External standard comparator, not a downstream outcome. |
| Language probe accuracy | Detects exploitable language identity | Diagnostic instrument; high/low values are not task performance. |
| Kernel-minus-linear CKA gap | Nonlinearity diagnostic | Instrument in the mapping battery. |
| Standard clustering indices | Proposed external comparators | Deferred until exact implementation contracts are verified. |

### 3.9 Frozen RMFS v1: historical scalar stress test

Frozen RMFS v1 combined:

- `qT`: normalized behavioral transfer from a representation intervention harness;
- `qL = 1 - LFS-VC`: REML concept-dominance direction;
- `qK = 1 - selected_mapping_rung/5`: mapping simplicity;
- `C_pres`: preservation in intervention mode.

The observational scalar was a soft minimum with frozen temperature 0.1. It failed its preregistered crash test and did not improve on LFS when expanded from four to fourteen models. The `qT` component was health-inverted and controlled many minima.

Status: **retired as a headline scalar; retained as a negative-result stress test**.

### 3.10 Profile 0.2: separate readings, no scalar

| ID | Reading | Computation | Availability |
|---|---|---|---|
| R1 | Concept dominance | `1 - direct-SS LFS` after joint coordinate standardization | Any balanced grid |
| R2 | Mean alignment margin | Macro mean of true-pair minus hardest-negative margins | Any pivoted parallel grid |
| R3 | Weak-language tail | Mean of the lowest 25% of per-language AaR@10 values | Any pivoted parallel grid |
| R4 | Preservation | Raw within/shared concept and effective-rank ratios against a frozen reference; legacy CVP gate at 0.90 | Intervention/checkpoint comparisons only |

Status: **computed and retrospectively analyzed**. The 81-checkpoint Profile 0.2 campaign is complete. The paired study/seed analysis is recorded in `docs/AIM1_PROFILE_AXES_RESULTS_2026-08-24.md`. The profile must be interpreted component by component. It is not a new RMFS scalar.

## 4. Dataset registry and exactly how each dataset was used

| Dataset/source | Support used | Aim 1 role | Important constraint |
|---|---:|---|---|
| **NTREX-128 / WMT newstest2019** | Up to 128 languages; 1,997 sentences in 123 documents | Primary matched-concept representation grid | English is the source and other sides are translations. New train/test splits must purge document IDs. |
| NTREX development slice | Usually first 300 sentences per language | Layerwise LFS, MEXA, AaR, model grid, many batteries | Absolute values depend on `L`, `N`, pooling, and standardization. |
| NTREX hub slice | 500 sentences | English-Hub donor controls | Larger `N` was used to reduce donor-ranking instability. |
| NTREX campaign slice | 1,500 sentences per language | Fourteen-model dumps, mapping budget, tail curves, whitening | Saved at four or five coarse layers per model; not a dense all-layer dump. |
| NTREX lens slice | 100 sentences; 11 non-Latin scripts | Corrected Logit Lens/script-mass experiment | Last-token and matched mean-pooling variants must not be mixed. |
| NTREX adjacent same-document pairs | About 100 pairs for behavior; 60 pairs for attention | Cross-lingual context transfer and attention | Matched context must be compared with same-language wrong-document context. |
| NTREX word-level derived data | Full available corpus; English plus 20 languages for alignment, narrower complete grids for LFS | Word-level reliability and LFS replication | Two aligners, content-word filtering, one-to-one intersection, and coverage reporting are mandatory. |
| **Belebele** | 300 four-choice questions per language; up to 122 languages | Main translated downstream target, zero-shot log-likelihood | Translated benchmark; chance floor 0.25; use one uniform prompt protocol. |
| **CulturaX** | Token-count proxy recovered for 95 languages in the original report | Main data-availability covariate | It is a proxy for undisclosed model training mixtures. |
| **CC-100 sizes** | Backstop for missing CulturaX values | Deflation robustness and low-resource coverage | Imputation must be flagged; bytes are calibrated to the token-count scale. |
| **Language metadata** | Macro-family and script labels for downstream languages | Deflation controls and matched-pair tests | Small families are grouped under a documented rule. |
| **Tokenizer fertility** | Per model-language tokens-per-character relative to English | Downstream covariate and pooling/fertility analysis | Fertility is both a confound and a representation-side phenomenon. |
| **Published MEXA tables/leaderboard** | Nine external models by 203 languages in the original anchor | External comparison and published-score anchor | External result, not project-owned data or metric. |
| **WMT19 bidirectional test sets** | de, ru, zh, fi, lt, gu, kk; 300 pairs in each available direction; 14 models | Native-language-side versus translationese-side measurement test | Same language/domain/year comparison; Czech excluded for lacking both directions. |
| **INCLUDE (`CohereForAI/include-base-44`)** | About 500 questions per language; 30 languages overlapping the panel; 14 models | Natively authored downstream exam | Questions are locally authored rather than translated; zero-shot log-likelihood. |
| **SimAlign/XLM-R plus eflomal alignments** | English to 20 NTREX languages across 11 scripts | Reliability-filtered word correspondences | Agreement failed the original 0.70 gate; intersection and segmentation corrections are required. |

Datasets not on the current Aim 1 critical path:

- TyDiQA-style targets were proposed historically, but no valid balanced-concept protocol was frozen.
- MGSM appears only as a possible secondary outcome for a future trained dissociation replication.
- Any dense-retrieval/RAG corpus belongs to deferred Aim 3 work and must not be started now.

## 5. Model inventory

### 5.1 Ten-model development grid

1. Qwen3-0.6B-Base
2. Qwen3-1.7B-Base
3. Qwen3-4B-Base
4. Qwen3-8B-Base
5. OLMo-2-0425-1B
6. OLMo-2-1124-7B
7. Mistral-7B-v0.3
8. EuroLLM-1.7B
9. BLOOM-1.7B (`bloom-1b7`)
10. BLOOM-7.1B (`bloom-7b1`)

Purpose: vary scale within families and multilingual design across families.

### 5.2 Held-out and added direct-SS models, bringing the historical cohort to 17

11. SmolLM2-1.7B
12. Falcon3-7B-Base
13. Salamandra-2B
14. Salamandra-7B
15. granite-3.1-8b-base
16. Yi-1.5-9B
17. Llama-3.1-8B

The later verification result is 17/17 direct-SS profiles with an interior minimum and endpoint recovery.

### 5.3 Fourteen-model saved-dump campaign

This is not the same as the 17-model direct-SS cohort. It contains:

1. EuroLLM-1.7B
2. Falcon3-7B-Base
3. Mistral-7B-v0.3
4. OLMo-2-0425-1B
5. OLMo-2-1124-7B
6. Qwen3-0.6B-Base
7. Qwen3-1.7B-Base
8. Qwen3-4B-Base
9. Qwen3-8B-Base
10. SmolLM2-1.7B
11. bloom-1b7
12. bloom-7b1
13. salamandra-2b
14. salamandra-7b

Purpose: `N=1500` saved representations for mapping, tail, whitening, Profile 0.2, Round 3 invariance, and anisotropy work.

### 5.4 Nineteen successful A9 expansion models

1. Falcon3-3B-Base
2. Llama-3.1-8B
3. Mistral-Nemo-Base-2407
4. OLMo-2-1124-13B
5. Qwen2.5-0.5B
6. Qwen2.5-1.5B
7. Qwen2.5-3B
8. Qwen2.5-7B
9. Sailor-1.8B
10. Sailor-7B
11. SmolLM2-360M
12. TowerBase-7B
13. Yi-1.5-9B
14. granite-3.1-8b-base
15. mGPT
16. occiglot-7b-eu5
17. xglm-1.7B
18. xglm-2.9B
19. xglm-7.5B

Three of these overlap the direct-SS cohort: Llama-3.1-8B, Yi-1.5-9B, and granite-3.1-8b-base. The verified accounting is therefore 33 unique models across the 14-model base plus 19-model expansion, or separate 17-model direct-SS and 19-model REML profile cohorts depending on the claim.

### 5.5 Special experiment subsets

| Experiment | Models |
|---|---|
| Lens and cross-lingual context transfer | Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, OLMo-2-1B |
| Attention mechanism | Qwen3-0.6B-Base, 60 pairs |
| Prototype mapping/battery | Qwen3-1.7B, OLMo-2-1B, BLOOM-1.7B |
| Pooling/fertility | Qwen3-0.6B, Qwen3-8B, OLMo-2-1B, BLOOM-1.7B, Salamandra-2B |
| Word-level LFS | Qwen3-1.7B, layer 13 |
| Translationese and native-exam audits | All 14 saved-dump campaign models |
| Round 3 invariance and anisotropy | All 14 saved-dump campaign models |
| Archived checkpoint/Profile 0.2 stress test | Qwen3-0.6B-Base reference plus 81 scientific checkpoints |

## 6. Complete Aim 1 experiment ledger

### Stage A: establish the representation measurement

| ID | Experiment | Models/data | Process | Primary readings | Status |
|---|---|---|---|---|---|
| A1 | Pilot metric suite | Qwen3-0.6B; 26-language NTREX pilot | Extract all layers, fp32 mean pool, standardize, compute LFS/MEXA/AaR/hub | LFS, AaR, Hub; MEXA baseline | **Verified complete** |
| A2 | Ten-model layerwise grid | Ten development models; up to 128 languages; `N=300` | Same pipeline at every layer; compare fingerprints, families, scales, resource tiers | LFS profile/dip, MEXA, AaR | **Verified complete** |
| A3 | Estimator floor and sample-size behavior | Synthetic grids plus monolingual pseudo-language controls and real nested samples | Compare observed LFS with analytic null over `L`, `N`, and `D`; compare LFS and retrieval movement as `N` changes | LFS null/excess; MEXA/AaR sensitivity | **Verified complete** |
| A4 | Direct SS versus REML parity | Fourteen shared models | Recompute both estimators on matched support | Direct LFS and LFS-VC | **Verified complete**: rank 1.000, max absolute difference 0.00498 |

### Stage B: interpret what the shared space contains

| ID | Experiment | Models/data | Process | Primary readings | Status |
|---|---|---|---|---|---|
| B1 | English-Hub and donor controls | Original grid; `N=500` | Cross-validated English versus latent-factor prediction plus individual donor ranking | Latent advantage, best donor | **Complete with correction** |
| B2 | Resource-tier and family fingerprints | Ten-model grid, later holdouts | Compare high/mid/low language alignment across scaling and recipe families | LFS dip, MEXA/AaR tiers | **Complete**, descriptive rather than causal |
| B3 | Mapping-complexity budget | Three prototypes then 14-model campaign | Cross-fitted M0-M5 ladder, permutation nulls, selected and common ranks | Offset/scale/rotation/linear/nonlinear shares | **Complete with document-purge correction** |
| B4 | Anisotropy and coordinate-axis audit | Fourteen models, 69 saved layers, 195,072 axes | Per-coordinate language shares against Gaussian/Beta reference, BY correction, spectrum comparisons | Coordinate shares, mean cosine, top-eigen share | **Complete**. Native-axis language effects are statistically widespread but energetically concentrated. The 14-model orthogonal-basis sensitivity shows widespread significance survives while energy concentration is strongly basis dependent. No causal-axis claim. |

### Stage C: validate against behavior and downstream performance

| ID | Experiment | Models/data | Process | Primary readings/targets | Status |
|---|---|---|---|---|---|
| C1 | Belebele zero-shot downstream panel | Ten models then 17; 300 questions/language | Uniform zero-shot log-likelihood; reject nonuniform 5-shot protocols | Accuracy, PaR@10 | **Verified complete** |
| C2 | Deflation/transfer-alpha analysis | Belebele + CulturaX + family/script/fertility | Fit model-FE regression, residualize, compare raw and deflated rank associations | MEXA/AaR versus alpha; later LFS comparison | **Complete with corrected uncertainty** |
| C3 | Statistical hardening | Existing panel | Family demeaning, disattenuation, floor robustness, fixed-depth sensitivity, family dropout, matched pairs, clustered bootstrap | Correlation bracket and metric differences | **Verified complete** |
| C4 | Native exam | INCLUDE; 30 languages; 14 models | Same zero-shot scoring and deflation as Belebele | Matching/LFS-family readings versus native accuracy | **Complete negative boundary**: N1/N3 pass; N2/N4 fail |
| C5 | Translationese direction | WMT19 both directions; seven languages; 14 models | Compare native-L and translated-L parallel grids | Matching, language share, tail | **Complete mixed result**: T1/T2 fail; T3/T4 pass |

### Stage D: test the proposal's Logit Lens and generation questions

| ID | Experiment | Models/data | Process | Primary readings | Status |
|---|---|---|---|---|---|
| D1 | Corrected Logit Lens | Qwen3 0.6/1.7/4B and OLMo-2-1B; 100 NTREX sentences; 11 scripts | Unembed per-layer states; measure script mass and max cosine to output embeddings; rerun matched pooling | Script-mass valley, lens validity | **Verified complete**; script and concept events occur at different depths |
| D2 | Cross-lingual context transfer, round 1 | Same four models; 13 context languages; about 100 adjacent pairs | English target NLL with foreign context versus no context | Context benefit | **Complete; underpowered threshold corrected prospectively** |
| D3 | Context transfer, round 2 | Same models; about 40 context languages | Add same-language wrong-document context; compute matched-minus-mismatched benefit; deflate both predictor and target | Content transfer versus alignment | **Verified complete**, but model support is only four |
| D4 | Attention mechanism | Qwen3-0.6B; 41 languages; 60 pairs | Eager attention, fp32 accumulation, matched-minus-mismatched context attention divided by uniform baseline | Content attention | **Complete**; alignment-attention holds, independent attention-behavior link is estimator-dependent/fails primary specification |

### Stage E: blind generalization and model expansion

| ID | Experiment | Models/data | Process | Primary reading | Status |
|---|---|---|---|---|---|
| E1 | Frozen walk-forward | SmolLM2-1.7B and Falcon3-7B | Apply the full frozen pipeline and exam-capable rule without retuning | Fingerprint and deflated validity | **Verified complete: 4/4 frozen predictions passed** |
| E2 | Salamandra governance holdout | Salamandra 2B/7B | Frozen fingerprint, tier, scale, and hub predictions | LFS/MEXA/hub | **Complete mixed result**, including a failed monotonic-scaling prediction |
| E3 | Added families | granite 8B, Yi 9B, Llama 8B | Frozen capability/fingerprint predictions | LFS profile and Belebele capability | **Complete mixed result** |
| E4 | A9 expansion | 21 intended, 19 successful new models | Same 300-sentence/41-language measurement rule; REML LFS-VC profile; zero-shot translated/native exams where available | LFS-VC, matching, AaR, downstream panel | **Verified representation expansion**, 19/19 profiles with endpoint recovery |

### Stage F: construct-boundary and falsification experiments

| ID | Experiment | Models/data | Process | Primary readings | Status |
|---|---|---|---|---|---|
| F1 | Synthetic injection battery | Three prototypes, then campaign | Offset, scale, shared/per-language rotation, shear, interaction, warp, collapse, pairing permutation, identity | LFS, MEXA, AaR, ladder, probe, CKA, rank, CVP | **Verified complete with documented prediction failures** |
| F2 | Misalignment calibration | Same prototypes | Inject known rotation/scale and recover magnitude | Recovered angle/scale | **Verified complete** |
| F3 | Whitened-basis robustness | Fourteen models | ZCA-whiten each layer and recompute LFS/MEXA | Native versus whitened profile | **Complete failed prediction**: rank agreement 0.45 |
| F4 | Campaign tail curves | Thirteen reported models from 14-model campaign, `N=1500` | Retain sentence margins and compute AaR@1/5/10/20 | Tail shape and mean-tail dissociation | **Verified complete** |
| F5 | Word-level replication | Qwen3-1.7B, English plus aligned language subsets | Two-aligner links, segmentation correction, content-word intersection, balanced grid, reconstruction and null gates | Word-level LFS | **Complete with correction**; one operating point retains useful headroom |
| F6 | Pooling/fertility robustness | Five models | Mean, final-token, unit-norm mean, and length-matched pooling under frozen decision rules | LFS depth, components, MEXA/AaR, fertility relation | **Complete failed robustness gate**; rankings are more stable than magnitudes |

### Stage G: Round 3 invariance, correction, and diagnostic-family work

| ID | Experiment | Models/data | Process | Primary readings | Status |
|---|---|---|---|---|---|
| G1 | Uniform and diagonal rescaling | Fourteen-model fixed 13-language/300-concept panel | Evaluate raw and jointly standardized tracks under scalar and coordinate scaling | LFS, LFS-VC, MEXA, AaR, CKA, CVP, norms | **Verified complete**: 3,414/3,416 analytic cells passed; two epsilon-level cosine flags diagnosed |
| G2 | Controlled concept-rank projection | Same panel | Remove concept subspace rank while preserving other additive components where constructed | LFS, margins, MEXA, ranks, preservation | **Verified complete** |
| G3 | Bottom-tier language shrinkage | Swahili, Amharic, Zulu, Khmer within the fixed panel | Uniformly shrink only named languages on raw and standardized tracks | LFS, AaR/margin, preservation | **Verified complete** |
| G4 | Raw LFS correction | Authoritative 128-language/300-concept grid | Rebuild dip LFS, layer-0 LFS, depth, analytic null, and dominance label | Direct-SS LFS and components | **Verified complete**; dip values 0.644-0.938, all above 0.5 and null 0.2981 |
| G5 | Profile 0.2 on archived checkpoints | 81 scientific checkpoints, one frozen reference, three seeds where applicable | Compute R1-R4 separately and join transparent frozen controls | R1-R4 plus raw geometry | **Computed and retrospectively analyzed**; 21 profiles still lack joined outcomes and the join has no primary Belebele accuracy |
| G6 | Coarse anisotropy/axis analysis | Fourteen models, 69 layers | Coordinate language-share tests, spectrum summaries, descriptive layer correlations | Coordinate shares and anisotropy | **Computed, interpretation open** |

## 7. Remaining Aim 1 work, in the correct order

### R0. Finish analysis of already-computed Profile 0.2 and axis outputs

No new model run is needed.

1. Deduplicate shared base/control profiles before inferential analysis.
2. Compare arms within each study and seed rather than pooling unrelated studies.
3. Report paired seed changes and intervals.
4. Verify R4 detection against the frozen, pre-labeled collapse/reorganization cases.
5. Separate internal geometry agreement from downstream validity. MEXA and AaR are not independent downstream tasks.
6. Treat four/five-layer axis correlations as descriptive only.
7. Test sensitivity of coordinate findings to basis choice and the null model before making a neuron-axis claim.

Status: **complete**. Profile analysis job: CLSP `1761807`. Basis-sensitivity array/assembly: `1761838`/`1761870`. Result records: `docs/AIM1_PROFILE_AXES_RESULTS_2026-08-24.md` and `docs/AIM1_AXIS_BASIS_SENSITIVITY_RESULTS_2026-08-24.md`.

### R1. Expand the exact Aim 1 section 3.1.3 content-transfer target

This is the highest-value remaining new experiment because the strong legacy content-transfer point estimate uses only four independent models.

Required stages before launch:

1. Write and sign a dated Aim 1 addendum.
2. Freeze the 19 successful A9 models; no replacements after outcomes.
3. Recover and record exact cached model revision hashes.
4. Freeze the document-purged matched/mismatched context pairs, language set, generation settings, failure policy, and output schema.
5. Run a one-model integrity smoke test.
6. Run the model array.
7. Analyze with model bootstrap and crossed model-language uncertainty; language-only intervals are secondary.
8. Report all technical failures and the result regardless of direction.

Status: **complete and audited**. Protocol: `prereg/addenda/AIM1_R1_CONTENT_TRANSFER_PROTOCOL_2026-08-24.md`. Authoritative analysis job: `1765948`. Across 19 models and 13 families, the primary deflated `q_L` to `R_content` association is rho `0.508`, model-bootstrap 95% CI `[0.188, 0.707]`, p `0.003`. See `docs/AIM1_R1_CONTENT_TRANSFER_RESULTS_2026-08-26.md` for the full result, repairs, robustness analyses, and claim boundary.

### R2. Fair same-support frozen RMFS v1 expansion

Purpose: test the already-frozen RMFS v1 components on the same expanded Aim 1 content-transfer support, not invent a new metric.

1. Reuse `qL` where the estimator/grid matches.
2. Fit `qK` with document-purged M0-M5 splits and the frozen one-SE rule.
3. Run the frozen `qT` harness with identity-patch canaries.
4. Assemble the scalar at the original temperature 0.1 without retuning.
5. Report `qT`, `qL`, `qK`, and RMFS separately.
6. Compare against content transfer and downstream alpha on identical rows with language, model, and crossed intervals.
7. Preserve RMFS as a negative result if it fails again; no post-hoc replacement scalar.

Status: **required only if RMFS is included as a fully expanded paper comparison; not run**.

### R3. Larger-model trained collapse/dissociation replication

Purpose: determine whether the 0.6B collapse/reorganization finding generalizes to Qwen3-1.7B and Qwen3-4B while treating the checkpoints solely as Aim 1 measurement stress tests.

Proposed fixed design:

- matched control and collapse-inducing arm;
- three seeds;
- raw concept preservation, standardized LFS, Profile readings, rank/norm, and independent downstream task performance;
- Belebele primary; MGSM secondary only after prospective power analysis;
- paired arm/seed intervals and an explicit minimum detectable effect.

This involves new training even though its scientific purpose is Aim 1 validation. It therefore requires a separate protocol, compute estimate, and explicit approval. If "Aim 1 only" is intended to prohibit all new training, this experiment remains deferred.

Status: **optional/gated, not authorized**.

### R4. Paper assembly and claim audit

1. Create the anonymous nine-page paper skeleton.
2. Build a machine-readable claim ledger mapping every displayed number to code, input, job, and hash.
3. Use direct SS as primary and REML as robustness.
4. Separate observation, synthetic intervention, trained intervention, retrospective analysis, and preregistered confirmation.
5. Complete the formal literature audit before any novelty or priority claim.
6. Freeze figures and tables only after R0 and the decision on R1/R2/R3.

Status: **required, not complete**.

## 8. Mandatory execution stages for every remaining experiment

Every future Aim 1 experiment must have all of the following documented.

### Stage 0: question and claim contract

- Research question.
- Primary and secondary hypotheses.
- Metric direction and failure interpretation.
- Exact claim the result could support.
- Explicit statement of what it cannot support.

### Stage 1: preregistration and power

- Dated frozen protocol before outcome inspection.
- Unit of analysis and target of generalization.
- Sample size and power/minimum detectable effect.
- Primary uncertainty method.
- Multiplicity correction and stop rules.

### Stage 2: data lock

- Dataset version and hash.
- Language list, concept/sentence IDs, document IDs, and exclusions.
- Train/selection/held-out splits.
- Document purging where contextual adjacency exists.
- Translation direction and native/translated status.

### Stage 3: model lock

- Exact Hugging Face ID and cached revision hash.
- Model family, size, layer count, tokenizer, precision, device, and context limit.
- Frozen layer-selection rule.
- No silent substitute after outcomes.

### Stage 4: extraction

- Tokenization and maximum length.
- Pooling definition.
- fp32 accumulation.
- Saved representation dtype.
- Balanced-grid completeness check.
- Raw and standardized tracks where relevant.

### Stage 5: parity and canaries

- Reproduce a known LFS/MEXA cell from the prior pipeline.
- Check finite values and exact output schema.
- Verify reference hashes.
- Run identity/no-op controls.
- Abort on missing languages, checkpoints, or mismatched controls.

### Stage 6: metric computation

- LFS plus all raw factor components.
- AaR/margins where parallel retrieval is relevant.
- Raw norm/rank/preservation where collapse is in scope.
- Mapping/hub readings only under their frozen protocols.
- External baselines clearly labeled.
- No composite created after seeing outcomes.

### Stage 7: statistical analysis

- Paired analysis for interventions.
- Language-family clustering for language generalization.
- Model clustering for model generalization.
- Crossed model-language inference when both are claimed.
- Leave-one-family/model sensitivity where support allows.
- Missingness and technical failures reported.

### Stage 8: artifact and discrepancy record

- Code commit.
- Input/output hashes.
- Cluster job and array IDs.
- Environment and runtime.
- Result manifest and completeness JSON.
- Error/discrepancy ledger entry for any deviation.

### Stage 9: reporting gate

- Separate metric construct validity from downstream utility.
- Separate reorganization detection from demonstrated harm.
- State negative and null results at equal prominence.
- Use the narrowest defensible claim.
- Do not edit the main technical report until the result audit is complete.

## 9. Corrections that must carry into all new writing

1. Do not write "31/31 models." Use 17/17 direct SS and 19/19 REML separately; there are 33 unique models after three overlaps.
2. Do not call endpoint recovery a formally established smooth U-shape.
3. Do not say LFS is exactly independent of evaluation size. State the analytic null and compare fixed grids.
4. Do not say a low LFS alone establishes language-agnosticism or better capability.
5. Do not use the untraced 0.38-0.87 raw-LFS range. The authoritative 14-model direct-SS dip range is 0.644-0.938 on the 128-language/300-concept grid.
6. Do not universalize 124/127 latent-over-English. Name the model; the 17-model range is 113-124.
7. Do not present old unpurged nonlinear mapping shares as clean estimates.
8. Do not present the original word-level primary as fully clean; the function-word correction changes its interpretation.
9. Do not call MEXA an original metric or a downstream task.
10. Do not say RMFS beats LFS. Frozen RMFS v1 is currently a negative result.
11. Do not call Profile 0.2 "RMFS" without qualification, and do not average R1-R4.
12. Do not claim representation collapse necessarily harms behavior. CVP detects reorganization; task harm requires independent evaluation.
13. Do not make a first-of-kind claim before the formal literature review.

## 10. Current completion gate

Aim 1 is ready for paper freeze only when:

- the Profile 0.2 and anisotropy outputs have a controlled written analysis;
- the exact paper claim is chosen;
- R1 content-transfer expansion is either completed or explicitly removed from the submission claim with Professor Koehn's agreement;
- the role of frozen RMFS v1 is fixed as stress test/negative result;
- any larger-model collapse replication is either preregistered and completed or explicitly deferred;
- every headline number resolves to code, immutable input, job, and hash;
- the literature/novelty audit is complete;
- no Aim 2 or Aim 3 result is implied.
