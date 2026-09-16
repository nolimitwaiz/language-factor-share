# Aim 1 R1: 19-Model Content-Transfer Expansion

Date: 2026-08-26  
Status: complete and audited  
Frozen protocol: `prereg/addenda/AIM1_R1_CONTENT_TRANSFER_PROTOCOL_2026-08-24.md`  
Authoritative analysis job: `1765948` (`COMPLETED`, exit `0:0`, 49 minutes 35 seconds)

## 1. Question

The legacy Aim 1 section 3.1.3 analysis found a strong relationship between the LFS concept-direction reading and held-out multilingual content transfer, but it used only four models. R1 asked whether that relationship survives across the frozen 19-model A9 panel and 13 model families.

The candidate is `q_L = 1 - LFS-VC`, the concept-direction form already frozen in the A9 panel. This is an LFS reading with the direction reversed so that larger values mean greater concept dominance. R1 did not test frozen RMFS v1 or the four-reading Profile 0.2 as a scalar.

## 2. Design

- Dataset: NTREX `newstest2019`.
- Sampling unit: 100 adjacent English target/context pairs from 100 distinct documents.
- Context conditions: no context, matched same-document context, and length-matched different-document context.
- Context languages: 41, including English.
- Scoring: deterministic teacher-forced English target-token NLL in fp32; no sampling and no free generation.
- Frozen model panel: 19 models from 13 families.
- Scientific records: 8,300 sequences per model and 157,700 total per-example rows.
- Primary target: `R_content`, the matched-versus-mismatched context benefit in language L divided by the English benefit, when the English denominator is positive.
- Primary analysis: deflated Spearman association between `q_L` and `R_content`, controlling for language exposure, macro-family, script, and tokenizer fertility, with a model bootstrap.
- Robustness: crossed model-language bootstrap, model-family bootstrap, raw unnormalized content, model-level medians, leave-one-family-out analyses, and same-support MEXA/AaR external comparators.

All 19 models had positive English content effects and valid ratios for all 40 non-English scoring languages. The frozen A9 geometry panel overlapped with 34 languages per model, giving 646 common-support model-language rows.

## 3. Main findings

| Candidate | Target | Deflated rho | Model-bootstrap 95% CI | Two-sided bootstrap p |
|---|---|---:|---:|---:|
| `q_L = 1 - LFS-VC` | `R_content` | **0.508** | **[0.188, 0.707]** | **0.003** |
| MEXA, external comparator | `R_content` | 0.591 | [0.330, 0.818] | 0.0005 |
| AaR, external comparator | `R_content` | 0.443 | [0.093, 0.709] | 0.014 |
| `q_L = 1 - LFS-VC` | raw `content` | **0.620** | **[0.337, 0.769]** | **0.001** |
| MEXA, external comparator | raw `content` | 0.665 | [0.468, 0.805] | 0.0005 |
| AaR, external comparator | raw `content` | 0.471 | [0.168, 0.691] | 0.001 |

The primary R1 result is positive and its model-bootstrap interval excludes zero. This closes the main weakness in the legacy four-model result: the relationship generalizes across the frozen 19-model panel on this fixed NTREX content-transfer harness.

The expanded effect is smaller than the legacy point estimate of 0.842. The correct interpretation is not that R1 reproduced the exact magnitude. It reproduced the direction and established a moderate association with model-level uncertainty that excludes zero.

MEXA has a larger point estimate than `q_L` on this harness. R1 therefore does not support a claim that LFS outperforms every existing alignment metric. It supports the narrower and useful claim that the LFS concept-direction reading carries reproducible information about cross-lingual content transfer while measuring a different, explicitly variance-partitioned property.

## 4. Robustness

For the primary `q_L` to `R_content` association:

- crossed model-language 95% CI: `[0.181, 0.723]`, p = `0.004`;
- model-family-bootstrap 95% CI: `[0.242, 0.757]`, p = `0.0005`;
- leave-one-family-out rho range: `0.439` to `0.598`; every estimate remained positive;
- model-level Spearman against median `R_content`: `0.770`, 95% CI `[0.440, 0.941]`, p = `0.0005`;
- model-level Spearman against median raw content: `0.786`, 95% CI `[0.472, 0.931]`, p = `0.0005`.

Raw content gives the same conclusion without the English normalization. Its leave-one-family-out `q_L` range is `0.544` to `0.675`. This matters because it shows that the result is not created only by dividing through the English content effect.

## 5. Claim boundary

The defensible paper claim is:

> Across 19 frozen models from 13 families, the LFS concept-direction reading was moderately associated with held-out multilingual content transfer on a preregistered NTREX harness after controlling for exposure, family, script, and tokenization, with model-, crossed-, and family-bootstrap intervals excluding zero.

R1 does not establish universal downstream validity, causal geometry-to-performance effects, retrieval quality, instruction-following quality, or Aim 2/Aim 3 performance. It also does not rescue direct LFS minimization as a training objective. MEXA and AaR remain external comparators; they are not project-original metrics.

## 6. Technical execution record

- Data lock: `1761872`.
- Sealed smoke: `1761877`, complete.
- Original array: `1761878`. Nine models completed; ten hit a pre-scoring tokenizer-length ceiling.
- Documented 1,024-token technical repair: `1765008`. Eight repaired models completed; Yi-1.5-9B and Mistral-Nemo still exceeded the ceiling.
- Documented 2,048-token final repair: `1765050`. Both remaining models completed. No examples, models, revisions, scoring equations, or scientific thresholds changed, and no sequence was silently truncated.
- Analysis attempts `1765051` and `1765947` failed before scientific output because of, respectively, a Slurm accounting date-window query and a duplicated directory component in the panel path.
- Authoritative analysis: `1765948`, complete with zero stderr.
- Final audit: 19 part manifests, 157,700 of 157,700 expected rows, 2,000 bootstrap replicates, and all eight output hashes verified both on CLSP and after local synchronization.

Canonical artifacts: `results/round3/r1_content_transfer_analysis_2026-08-24/`.

## 7. Immediate next action

R1 is complete. Update the Aim 1 paper from the legacy four-model wording to the bounded 19-model claim above, create the corresponding table/figure, and review the wording with Professor Koehn before beginning any Aim 2 or Aim 3 campaign.
