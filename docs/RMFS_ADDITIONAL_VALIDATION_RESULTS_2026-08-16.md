# RMFS additional validation results, 2026-08-16

## Scope and provenance

The analysis follows `docs/RMFS_ADDITIONAL_VALIDATION_PLAN_2026-08-16.md`.
It does not change the frozen RMFS v1 definition, select a new scalar, or use
MEXA as a project metric.

- CLSP job `1753884`: completed in 14 minutes 14 seconds, exit 0.
- Safety gates: 105 of 105 RMFS tests and 5 of 5 Profile tests passed.
- RMFS v1 uncertainty: 2,000 bootstrap requests per dependence axis, seed 13.
- Axes: language, model, and crossed model by language.
- Alpha support: 522 rows, 14 models, 40 languages.
- Content-transfer support: 128 rows, 4 models, 34 languages.
- CLSP job `1753887`: completed in 14 minutes 23 seconds, exit 0. This is the
  final RMFS v1 robustness table because it reports requested and finite
  bootstrap counts separately. Seventy-two of 78 rows retained all 2,000
  replicates; six four-model content rows retained 1,967 to 1,995 finite
  replicates after constant-input resamples were discarded.
- Job `1753884` remains authoritative for the independent Profile reliability
  table.

## 1. Frozen RMFS v1 does not generalize as an alpha predictor

All quantities point in the favorable direction, so higher is better.

| Candidate | Alpha rho | Language CI | Model CI | Crossed CI |
|---|---:|---:|---:|---:|
| qL, concept direction | 0.158 | [0.034, 0.285] | [-0.013, 0.285] | [-0.038, 0.368] |
| qT, behavioral transfer | -0.011 | [-0.114, 0.077] | [-0.191, 0.151] | [-0.243, 0.181] |
| qK, mapping simplicity | -0.043 | [-0.137, 0.049] | [-0.164, 0.081] | [-0.218, 0.143] |
| Frozen RMFS, tau 0.1 | 0.039 | [-0.045, 0.109] | [-0.138, 0.216] | [-0.177, 0.225] |
| RMFS without qT | 0.089 | [-0.033, 0.206] | [-0.093, 0.231] | [-0.130, 0.309] |
| RMFS without qK | 0.071 | [-0.007, 0.141] | [-0.111, 0.236] | [-0.156, 0.270] |
| Arithmetic component mean | 0.022 | [-0.073, 0.115] | [-0.138, 0.176] | [-0.195, 0.214] |

Interpretation: the language-only interval makes qL look positive, but the
model and crossed intervals include zero. The binding uncertainty is the
model dimension. The frozen scalar is null under every dependence treatment.

## 2. The failure is not a bad temperature choice

Across the fixed sensitivity grid `tau = 0.025, 0.05, 0.1, 0.2, 0.5, 1.0`,
alpha rho remains between 0.023 and 0.043. Every model and crossed interval
includes zero. The candidate rankings remain close to the frozen tau 0.1
ranking, with Spearman agreement from 0.935 to 1.000.

No temperature is selected. The result rules out a simple aggregation-
temperature rescue.

## 3. qT weakens content-transfer validity

| Candidate | Content rho | Language CI | Model CI | Crossed CI |
|---|---:|---:|---:|---:|
| qL | 0.842 | [0.790, 0.905] | [0.032, 0.918] | [0.129, 0.935] |
| qT | -0.266 | [-0.376, -0.111] | [-0.750, 0.375] | [-0.744, 0.430] |
| qK | 0.513 | [0.374, 0.738] | [-0.147, 0.740] | [-0.168, 0.858] |
| Frozen RMFS, tau 0.1 | 0.545 | [0.417, 0.642] | [-0.271, 0.803] | [-0.316, 0.807] |
| RMFS without qT | 0.744 | [0.673, 0.871] | [0.118, 0.857] | [-0.002, 0.908] |
| Arithmetic component mean | 0.602 | [0.520, 0.776] | [-0.040, 0.739] | [0.022, 0.817] |

The corrected finite-replicate calculation leaves the qL model and crossed
intervals above zero. This supports qL on the current content panel, but the
target still contains only four independent models, so it does not justify a
population-level claim. More importantly, qT has the wrong sign and pulls the
scalar away from the strongest component. The no-qT ablation is retrospective
and is not selected as a replacement metric.

## 4. Soft-min control mechanism

| Component | Controls raw minimum | Mean soft-min weight, tau 0.1 |
|---|---:|---:|
| qT | 54.8% | 46.5% |
| qK | 30.5% | 31.7% |
| qL | 14.8% | 21.9% |

The frozen scalar mostly follows qT even though qT is null against alpha and
negative against content transfer. This is the measured failure mechanism.

The leave-one-model-out alpha ranges reinforce the same conclusion:

- qL: 0.122 to 0.208;
- frozen RMFS: 0.005 to 0.092;
- RMFS without qT: 0.056 to 0.147;
- qT: -0.065 to 0.040.

## 5. Profile 0.2 readings are reliable measurements

This test used no downstream outcomes. It measures stability from the
300-sentence selection block to 1,200 held-out sentences, partitioned into
four disjoint 300-sentence blocks.

| Reading | Selection to held-out rho | Median block-pair rho | Minimum block-pair rho | ICC(1,1) |
|---|---:|---:|---:|---:|
| R1 concept dominance | 0.934 | 0.989 | 0.978 | 0.996 |
| R2 mean alignment margin | 0.974 | 0.978 | 0.960 | 0.994 |
| R3 weak-language tail | 0.960 | 0.886 | 0.793 | 0.988 |

R1 and R2 are extremely stable. R3 is also reliable in value, but individual
model ranks move more in the tail: maximum six rank positions across blocks,
versus two for R1 and three for R2. Tail estimates should therefore keep
block or bootstrap uncertainty beside them.

## 6. Decision

1. Do not submit or present RMFS v1 as a validated universal scalar.
2. Do not tune tau or choose an ablation after seeing these results.
3. Keep RMFS v1 as a rigorous negative result showing why plausible gauges
   should not automatically be fused.
4. Lead the Aim 1 submission with LFS. Its concept direction remains the
   strongest original component on both targets tested here.
5. Continue with the non-composite diagnostic profile only after freezing
   `prereg/PREREG_FAMILY_V0.md`; high measurement reliability does not yet
   establish downstream validity.
6. Do not spend the estimated eight-plus GPU-hours on a 19-model original-
   RMFS extension before the Wednesday presentation. The present failure is
   mechanistic, not merely low power.
