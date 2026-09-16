# Aim 1 Profile 0.2 and Coordinate-Axis Results

Date: 2026-08-24  
Status: retrospective descriptive analysis complete; coordinate-basis sensitivity completed separately  
Authoritative CLSP job: `1761807` (`COMPLETED`, exit `0:0`, 8 seconds)  
Analysis contract: `docs/AIM1_RETROSPECTIVE_PROFILE_AXES_ANALYSIS_2026-08-24.md`

## 1. What was analyzed

Two already-computed Aim 1 artifacts were analyzed without running another model:

1. Profile 0.2 on 81 scientific checkpoint profiles. Seventy-two study/seed/arm rows had joined outcomes; 21 unique checkpoint profiles had no joined outcome row.
2. The coordinate-axis audit covering 14 models, 69 saved layers, and 195,072 native hidden-state axes.

The four intervention studies were kept separate. Every intervention was compared with the matched control from the same study and seed. The frozen Qwen3-0.6B base checkpoint appears in three study tables; it was centered within each complete study and averaged into one pooled analysis unit rather than counted three times. The pooled descriptive associations therefore contain 18 independent non-control analysis units, not 20 rows.

This is a retrospective analysis because the outcomes were visible before the contract was frozen. The associations below are descriptive effect summaries, not confirmatory estimates.

## 2. Profile 0.2 findings

### 2.1 The frozen R4 contrast works on the pre-labeled reorganization case

The pre-specified word-alignment contrast was control `WA-B`, collapse/reorganization arm `WA-C`, and guarded arm `WA-G`, with three seeds per arm.

| Arm | Within-concept ratio, mean | Shared-concept ratio, mean | Effective-rank ratio, mean | Legacy preservation gate |
|---|---:|---:|---:|---:|
| WA-B control | 1.011 | 1.013 | 0.938 | 3/3 pass |
| WA-C collapse/reorganization | 0.086 | 0.058 | 5.555 | 0/3 pass |
| WA-G guarded | 1.011 | 1.012 | 0.932 | 3/3 pass |

This is a clean construct-validity result for R4 on the frozen contrast. R4 distinguishes severe loss of within-concept and shared-concept structure from a guarded arm whose values remain near the frozen reference.

It does **not** by itself prove behavioral harm. It detects geometric reorganization or collapse; independent task evaluation is still required to establish harm.

### 2.2 The profile separates information that a scalar LFS cannot

Across the 18 independent non-control arm units, after centering both variables within study:

| Predictor | Comparison target | Spearman rho | Interpretation |
|---|---|---:|---|
| R2 mean-alignment margin | AaR change | 0.917 | Strong convergent agreement between two related alignment readings |
| R3 weak-language tail | AaR change | 0.872 | Strong convergent agreement, including sensitivity to weak-language behavior |
| LFS change | AaR change | 0.841 | Higher, not lower, LFS accompanied higher AaR in these trained arms |
| R4 effective-rank ratio | Favorable perplexity change | -0.837 | Rank inflation/reorganization accompanied worse perplexity |
| R4 shared-concept ratio | Favorable perplexity change | 0.536 | Preserved shared energy was moderately associated with better perplexity |
| R4 within-concept ratio | Favorable perplexity change | 0.529 | Preserved within-concept structure was moderately associated with better perplexity |

The R2-to-AaR result remained positive under every leave-one-study-out analysis (`rho=0.837` to `0.965`). The R3-to-AaR result also remained positive (`rho=0.745` to `0.973`). These are convergent comparisons, not downstream validation, because R2, R3, and AaR are built from related parallel-alignment structure.

The LFS sign is scientifically important. Profile R1 is exactly `1 - LFS`, so R1 and LFS contain the same information with opposite orientation. In this intervention set, lower LFS was not a reliable health objective: the pooled LFS-to-AaR association was positive (`rho=0.841`). This is consistent with the known shortcut failure in which a ratio can improve while useful concept structure is damaged or reorganized.

The R4 effective-rank association with favorable perplexity was negative in every leave-one-study-out analysis (`rho=-0.921` to `-0.727`). This supports using raw preservation/rank readings as a safeguard around LFS. It is still retrospective, based on a small number of arms from one base-model family, and must be confirmed prospectively before it becomes a general predictive claim.

The trained- and held-out-context deltas had weak or inconsistent associations with the profile. That is not treated as evidence against or for the profile because the earlier intervention work showed that context reliance can increase when representations are damaged.

### 2.3 What is missing

Twenty-one checkpoint profiles lack joined outcomes:

- `ADV` and `ADV-CVP`, three seeds each;
- the 2,000-step code-switch control and four 2,000-step intervention arms, three seeds each.

The joined file also does not contain the primary Belebele accuracy outcome. Therefore this analysis cannot establish that Profile 0.2 predicts downstream accuracy. MEXA and AaR cannot substitute for that test.

## 3. Coordinate-axis findings

The native-axis result has two simultaneous parts:

1. Language share is statistically diffuse across coordinates. The median layer's median coordinate language share was `0.874`. Sixty-five of 69 layers marked every coordinate language-heavy under the Benjamini-Yekutieli correction, and even the minimum layer fraction was `0.9963`.
2. Language **energy** is strongly concentrated. The median Gini coefficient of coordinate `SS_language` was `0.855`, and the top 5% of raw-variance axes carried a median `79.6%` of total coordinate `SS_language`.

The median within-layer Spearman correlation between coordinate language share and raw variance was `0.692`. Thus, almost every native coordinate has a language share far above the isotropic Gaussian reference, while a small high-variance subset carries most of the absolute language-effect energy.

This does not mean that nearly every coordinate is an independently causal "language neuron." Coordinate shares are basis dependent, hidden dimensions can be rotated without changing model function, and the exact Beta reference holds only under the stated independent isotropic Gaussian null. The defensible current statement is:

> In the models' native coordinate bases, systematic language main effects are statistically widespread but energetically concentrated.

The completed random orthogonal-basis sensitivity experiment is recorded in `docs/AIM1_AXIS_BASIS_SENSITIVITY_RESULTS_2026-08-24.md`. It found that widespread statistical language effects survive the rotations, while native-axis language-energy concentration falls sharply. The coordinate view may therefore appear only as a basis-sensitivity result, not as an identifiable-neuron claim.

## 4. Scientific conclusion

The evidence favors a diagnostic family over either LFS alone or a replacement scalar:

- LFS remains a useful global variance-partition diagnostic.
- R2 and R3 expose mean and weak-language alignment behavior that the LFS ratio does not isolate.
- R4 detects preservation failure and rank reorganization that can make a low LFS misleading.
- The readings must remain separate. Averaging them would hide the very disagreements that revealed the failure mode.

This result does not revive frozen RMFS v1. RMFS v1 remains a negative scalar result. Profile 0.2 is a separate four-reading diagnostic profile and should not be renamed or reported as a single RMFS score.

## 5. Reproducibility record

Frozen input hashes:

| File | SHA-256 |
|---|---|
| `profile_by_checkpoint.csv` | `f3b59e34d27f56bdc68bfd8b972204acaeac5131062fc7817f7c846d5cd0ef0c` |
| `profile_with_outcomes.csv` | `87be72cdaea47a96c47cd22601c52ceb14423564303c5d71e506d6e36f37525c` |
| `layers.csv` | `f426560d8166de22cb0a2277fc0b7df58f371dc7ea0d9971fd56152bbfe921d4` |
| `axes.csv` | `f1890a9d95c80df0445a3e89b96e1a852e4b75b69d69bc78f713f6b402c34f2e` |

Final output hashes:

| File | SHA-256 |
|---|---|
| `analysis_summary.json` | `e89f14f77cce89558c0cf4ff4fe46d8cf1f20594db34de9ff0250cbf4a83ed79` |
| `profile_associations.csv` | `cac4d976706555bc2bc034fab43486bf68813e587bd167152934f951ec1323d3` |
| `profile_arm_level.csv` | `213c7ae846c2607a4e8ea244f705bf073fdef162f94490c01acd2e6ebdf46a3e` |
| `profile_paired_arm_summary.csv` | `05f12b6d83a5078a7365ac23f65e0c92e0de1da75584a619e77f2cb5add40cb1` |
| `profile_r4_frozen_contrast.csv` | `4e3ffdce5418b064c59704c3d43fea11199a75f1e09914fd801a4c284b5ac2a1` |
| `profile_missing_outcomes.csv` | `acebecb77de4b0843f1a49f8e427fd1230ce9d4426511bd6571211cd76bad26b` |
| `axis_layer_summary.csv` | `adb512e3d42991c20faefc712e2d41bce9e2c6db68b731de09fdace695eef1d1` |
| `axis_model_minimum_summary.csv` | `17f685d468eaf85b8203c09b0d096ae4caeaba169e78e307d4e75807dfd1a582` |

The initial job `1761801` exposed duplicated base rows in the pooled analysis. Job `1761802` deduplicated before study centering, which made study-specific comparisons incomplete. The authoritative job `1761807` preserves complete within-study comparisons, centers within study, and then averages repeated frozen-base rows into one pooled unit. Only `1761807` is reportable.

## 6. Immediate next action

R0 is complete. Do not start Aim 2 or Aim 3. Choose whether the paper's next new experiment is the 19-model content-transfer expansion (R1) or whether that claim is removed with Professor Koehn's agreement.
