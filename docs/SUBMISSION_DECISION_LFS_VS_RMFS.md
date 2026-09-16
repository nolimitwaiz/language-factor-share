# Submission decision: LFS versus RMFS

**Date:** 2026-08-16  
**Decision:** Submit LFS as the core contribution. Use RMFS as a stress-test
campaign and extension, not as the headline metric.

## Why this is the correct decision

Professor Koehn gave the green light to the LFS work. The empirical record
also supports LFS over the preregistered RMFS scalar.

LFS has a simple, interpretable construct: the share of systematic additive
variation associated with language rather than language plus concept. It has
broad layerwise evidence, repeated U-shaped fingerprints, estimator
robustness, and behavioral associations. Its limitations are mathematically
understandable and experimentally demonstrated.

RMFS v1 was a serious attempt to guard those limitations. It improved the
research by showing that no single scalar survived every test. However, the
soft-min scalar failed its preregistered crash test, its four-model downstream
association fell to nearly zero on the 14-model grid, and one component was
health-inverted. Those are publishable negative findings, not grounds to
replace LFS in the submission.

## What was compared on 2026-08-16

The comparison reused already scored Aim 1 artifacts. It did not train or load
a model. It is a retrospective, same-support synthesis and must not be called
a new preregistered experiment.

Every candidate was oriented so higher is more favorable:

- Original LFS direction: `1 - direct-SS LFS`.
- REML LFS-VC direction: `q_L = 1 - LFS-VC`.
- RMFS v1: preregistered observational soft-min of `q_T`, `q_L`, and `q_K`.
- RMFS without T: soft-min of `q_L` and `q_K`.
- MEXA: published external comparator under shared preprocessing.
- AaR: project tail diagnostic.

The statistics use the original deflation controls for training tokens,
language family, script, and fertility. The same rows are used for each score
within a target.

## Results

### Downstream alpha

Support: 522 rows, 14 models, 40 languages.

| Reading | Deflated Spearman | Language-cluster 95% CI | Model-cluster 95% CI |
|---|---:|---:|---:|
| Original LFS direction | 0.158 | [0.045, 0.294] | [-0.023, 0.290] |
| REML LFS-VC direction | 0.158 | [0.045, 0.294] | [-0.023, 0.290] |
| RMFS v1 | 0.039 | [-0.045, 0.107] | [-0.137, 0.212] |
| RMFS without T | 0.089 | [-0.030, 0.206] | [-0.095, 0.229] |
| MEXA, external | 0.167 | [0.094, 0.228] | [-0.068, 0.407] |
| AaR | 0.006 | [-0.043, 0.079] | [-0.124, 0.212] |

The point difference `RMFS v1 - LFS` is -0.119. The language-cluster
interval is [-0.262, -0.016], but the model-cluster interval is
[-0.315, 0.078]. The honest conclusion is that LFS is stronger on this frame,
while 14 models still leave model-level uncertainty.

### Legacy held-out content transfer

Support: 128 rows, 4 models, 34 languages.

| Reading | Deflated Spearman | Language-cluster 95% CI | Model-cluster 95% CI |
|---|---:|---:|---:|
| Original LFS direction | 0.842 | [0.792, 0.903] | [0.024, 0.909] |
| REML LFS-VC direction | 0.842 | [0.792, 0.903] | [-0.019, 0.909] |
| RMFS v1 | 0.545 | [0.420, 0.645] | [-0.271, 0.723] |
| RMFS without T | 0.744 | [0.675, 0.866] | [0.078, 0.857] |
| MEXA, external | 0.769 | [0.692, 0.848] | [0.304, 0.900] |
| AaR | 0.680 | [0.575, 0.777] | [0.345, 0.828] |

The point difference `RMFS v1 - LFS` is -0.297. The language-cluster
interval excludes zero, but the model-cluster interval crosses zero because
only four models have the legacy target. The paper must state this limit.

## How the prior RMFS findings fit

- S1 crash test: failed in all three seeds.
- Strict synthetic battery: failed overall; `C_pres` detected the designated
  collapse cells exactly.
- Four-model alpha association: did not generalize to 14 models.
- Behavioral transfer T: health-inverted across three study designs.
- LFS structure channel: strong relation to held-out content transfer on its
  measured support.
- Fault-containing intervention panel: AaR, not RMFS structure, best tracked
  downstream harm.
- Severe measured-layer reorganization was not harmful on the tested 0.6B
  Belebele setting.

The resulting lesson is not "RMFS failed, therefore discard it." The lesson
is that a transparent family of readings is more scientifically defensible
than an aggregate score whose components succeed on different targets.

## Submission framing

A safe working title is:

**Language-Factor Share: Measuring and Stress-Testing Multilingual
Representation Structure**

Suggested paper structure:

1. Motivation from the NSF Aim 1 question.
2. LFS mathematical definition and estimator specification.
3. Broad layerwise fingerprints across models and languages.
4. Mapping, hubness, and behavioral validation.
5. Confound control, held-out tests, and uncertainty.
6. Synthetic and trained intervention stress tests.
7. RMFS attempt and why the scalar did not survive.
8. Practical recommendation: LFS plus raw components, tails, preservation,
   and downstream task checks.

## What must be fixed before submission

1. Pick one primary LFS estimator. **Locked 2026-08-17:** original
   committed direct-SS implementation, with REML LFS-VC as robustness
   (`docs/ENGINEERING_DECISIONS_2026-08-17.md`).
2. State the grid, pooling, standardization, layer-selection, and residual
   treatment for every headline LFS number.
3. Report raw language and concept components with the ratio.
4. Keep MEXA visibly labeled as an external baseline.
5. Retire the post-hoc RMFS definition that absorbed MEXA and AaR from the
   active narrative. Preserve its addendum only for provenance.
6. Label the new same-support comparison retrospective.
7. Use model-cluster uncertainty alongside language-cluster uncertainty.
8. Rebuild all paper tables and figures from exact source artifacts.
9. Complete a formal literature review before a novelty or priority claim.
10. Do not start a duplicate Aim 1 model grid.

## Optional next experiment

The only high-value Aim 1 scale extension is to run the established content
transfer harness on the 19 newly profiled models, expanding model support
toward 33. It would directly address the current four-model limitation. This
requires a separate compute estimate and explicit approval if it exceeds one
GPU-hour. It is not necessary to decide whether the current submission should
lead with LFS.

## Reproduction

- Script: `rmfs/scripts/compare_lfs_rmfs_submission.py`
- Manifest: `rmfs/results/submission_comparison/manifest.json`
- Estimates: `rmfs/results/submission_comparison/same_support_estimates.csv`
- Paired differences:
  `rmfs/results/submission_comparison/paired_differences.csv`

