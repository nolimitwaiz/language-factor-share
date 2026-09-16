# Round 3 Execution Protocol: Invariance, Correction, and Dissociation

**Date fixed:** 2026-08-24  
**Owner authorization:** Waiz Khan approved proceeding after reviewing the
corrected plan in the Codex research thread.  
**Status:** fixed before the new Round 3 result tables are computed  
**Scope:** diagnostic evaluation and already-authorized sub-one-GPU-hour RMFS
checkpoint audit; no change to frozen RMFS v1

## Research question

Which multilingual representation readings are invariant to harmless
reparameterizations, which detect destructive loss of scale or rank, and which
predict task-facing damage? The purpose is not to produce a universal score.
It is to state the construct boundary of each reading and test whether a
purpose-specific family is more informative than any scalar aggregate.

## Fixed reporting family

1. Direct sums-of-squares LFS and its raw language, concept, and residual
   components.
2. RMFS Profile 0.2 readings reported separately: concept dominance, mean
   parallel margin, weakest-language AaR@10, and raw preservation against a
   frozen reference.
3. AaR as the project tail diagnostic.
4. CVP, norm, and rank as raw preservation diagnostics.
5. MEXA, CKA, and standard clustering indices only as external comparators.

No 0-to-100 composite will be created. MEXA will not be described as a project
metric.

## Phase 0: execution repair

1. Freeze the 81 scientific Aim 2 checkpoints in an explicit manifest.
2. Exclude both adversarial smoke checkpoints.
3. Use the completed frozen base grid from job `1759003`.
4. Map frozen outcome controls `A` and `WA-A` to the base-model profile during
   the join rather than inventing checkpoint directories.
5. Require all 81 scientific result files and one identical reference-grid
   hash before producing a joined table.

## Phase A: invariance and correction battery

All perturbations are evaluated on two tracks:

- raw hidden states;
- the standard joint per-coordinate-standardized evaluation pipeline.

Perturbations:

1. Uniform positive shrinkage.
2. Positive per-coordinate shrinkage.
3. Controlled rank projection.
4. Per-language shrinkage.
5. Anisotropic synthetic controls.

Fixed analytic expectations:

- Uniform positive scaling leaves standardized direct-SS LFS unchanged and
  leaves exact cosine readings unchanged, up to documented numerical
  tolerance. Raw quadratic energies and CVP change by the square of scale.
- Positive diagonal scaling is removed by joint coordinate standardization
  when every retained coordinate has nonzero variance. Raw scale-sensitive
  readings remain allowed to change.
- Per-language scaling is not generally removed by joint coordinate
  standardization.
- Rank projection should be detected by rank diagnostics; whether LFS, AaR,
  or CKA responds is an empirical construct-boundary question.
- Mapping-ladder rung `K` is not declared exactly scale-invariant because the
  current implementation contains fixed ridge penalties. It is reported as an
  implementation sensitivity test.

## Phase B: corrected LFS reporting

For every available model and layer, report:

- raw LFS;
- minimum LFS and dip depth;
- analytic balanced-grid Gaussian null and excess over that null;
- language, concept, and residual sums of squares;
- raw norms and preservation components;
- per-coordinate language shares with synthetic and permutation calibration;
- both singular-value entropy rank and covariance-eigenvalue entropy rank,
  named separately;
- participation ratio.

LFS below 0.5 is interpreted as concept main-effect dominance over language
main-effect variance. The report will not claim language dominance for such a
cell.

## Phase C: trained dissociation replication

The word-alignment collapse experiment will be repeated on Qwen3-1.7B and
Qwen3-4B with three seeds and matched controls, subject to a separate compute
estimate before submission.

Fixed directional predictions:

1. Raw concept preservation falls substantially in the collapse arm.
2. Standardized LFS changes much less than the raw collapse magnitude and is
   not expected to improve.
3. Different profile readings disagree because they measure different failure
   modes.
4. Behavioral effects are decided with confidence intervals and a stated
   minimum detectable effect, not by the sign of a noisy point estimate.
5. Belebele is primary; MGSM is secondary unless a prospective power analysis
   supports a different choice.

The existing 0.6B observation motivating this replication is descriptive, not
a new prediction: mean word-alignment LFS changed only slightly while raw CVP
fell to roughly five percent of reference.

## Phase D: direct downstream utility

A three-model retrieval pilot will compare dip, middle, and final layers on a
real multilingual retrieval task, with an external retrieval or translation
baseline. This precedes a full RAG claim. TyDiQA will not be used as a parallel
hub-replication set unless a valid concept-alignment protocol is first fixed.

## Inference

Uncertainty follows the target of generalization:

- language-family clustering for language generalization;
- model clustering for model generalization;
- crossed model-language procedures where both are claimed;
- paired seed and arm comparisons for interventions.

No single clustering scheme replaces all confidence intervals.

## Stop rules

- Stop the RMFS join if any scientific checkpoint result is missing, reference
  hashes disagree, a smoke checkpoint enters the manifest, or a frozen control
  cannot be matched transparently.
- Do not launch any new model-training campaign exceeding approximately one
  GPU-hour without a separate estimate and owner approval.
- Do not edit the main technical report until the result audit is complete.
- Preserve null and negative results without renaming or redefining a metric
  after inspection.
