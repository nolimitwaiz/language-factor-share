# Round 3 Phase A Core Results

**Date:** 2026-08-24  
**Status:** complete for the 14-model core invariance, nonuniform-degradation,
and raw-LFS correction analyses  
**Main paper:** not edited

## Provenance

- Frozen model manifest: 14 saved campaign models, SHA-256
  `5d9ebb38115538173fcabe86485f935271d24ed37b3e9e0ea434ec796a0720f4`.
- Core CPU array: CLSP job `1761595`, 14 of 14 tasks completed.
- Completeness and analytic-scorecard join: job `1761603`, completed.
- Raw-LFS correction job: `1761622`, completed.
- Post-result descriptive summary: `1761633`, completed and explicitly
  labeled non-confirmatory.
- Core sweep: 7,448 metric cells.
- Prospective analytic checks: 3,416.
- No composite RMFS score was computed.

## Evaluation object

The invariance sweep used each model's saved pre-normalization dip-layer dump,
English plus 12 fixed target languages, and the first 300 aligned NTREX
concepts. Every perturbation was evaluated on two tracks:

1. raw hidden states;
2. joint per-coordinate-standardized hidden states.

The raw/depth correction table is a different, previously established grid:
all 128 languages and 300 concepts from the authoritative
`results/grid/<model>/metrics.json` artifacts. Values from these two panels
must not be merged or compared as though their language grids were identical.

## Headline invariance result

Of 3,416 prospective analytic cells, 3,414 passed at absolute tolerance
`1e-9` plus relative tolerance `1e-6`.

The only two flags occurred for Mistral-7B-v0.3 at raw scale `m=0.01`:

- AaR@10 moved by `1.11e-8`.
- Mean alignment margin moved by `6.71e-9`.

The external cosine helper normalizes by `norm + 1e-9`. At extreme shrinkage,
that fixed epsilon changes the normalized vector by a tiny amount. The flags
are therefore diagnosed numerical non-invariance, not evidence against the
underlying cosine-scale identity.

All of the following held across the 14 models within tolerance:

- direct-SS LFS is invariant to uniform positive scale;
- REML LFS-VC is invariant to uniform positive scale;
- MEXA's discrete retrieval result is invariant to uniform scale;
- linear CKA is invariant to uniform scale;
- raw concept-preservation energies scale quadratically;
- raw norms scale linearly;
- joint coordinate standardization removes uniform and positive diagonal
  scaling from every downstream reading in the tested pipeline.

This supports the paper's central class-level result: several widely used
ratio, cosine, order, and normalized-kernel readings cannot diagnose pure
representation shrinkage. A raw scale-sensitive preservation channel is not
optional if collapse is in scope.

## Nonuniform degradation results

### Positive diagonal shrinkage

After coordinate standardization, all 840 analytic diagonal-scaling cells
passed. The largest deviations were numerical rounding. This directly confirms
that the implemented LFS pipeline is invariant to the full positive diagonal
group, not merely to scalar shrinkage.

On the raw track, shrinking 25, 50, or 90 percent of coordinates by 0.1 caused
substantial energy loss. Median raw CVP and shared-concept preservation changes
across the diagonal cells were approximately -0.465 and -0.459, respectively.
Median direct-LFS movement was approximately -0.00007, although the model-level
range was wider. Thus the ratio often barely moves while raw capacity changes.

### Controlled concept-rank projection

The projection changed only the additive concept main effect; language and
residual components were preserved up to numerical precision on the raw track.
At the strongest tested projection:

- every model's AaR, mean margin, and MEXA decreased;
- every model's covariance-entropy rank, singular-entropy rank, and
  participation ratio decreased;
- standardized direct LFS increased in every model, with median delta `+0.133`;
- raw direct LFS increased only `+0.0024` at the median;
- standardized MEXA decreased by `-0.294` at the median;
- covariance-entropy rank decreased by about 91.6 percent at the median after
  standardization;
- participation ratio decreased by about 81.1 percent at the median after
  standardization.

Across the five retained-rank levels, covariance-entropy rank, participation
ratio, AaR, mean margin, MEXA, concept energy, and both preservation readings
were monotone with retained rank in all 14 models. Direct LFS was monotone in
the opposite direction: more concept-rank destruction raises the language
share.

Therefore the answer to the preregistered question is nuanced. The current
suite does detect this constructed rank loss, particularly after standardizing
and through retrieval behavior. Effective rank and participation ratio provide
the clearest reference-free diagnosis of the lost dimensional structure. Raw
LFS alone is weak for this failure mode.

### Bottom-tier per-language shrinkage

Uniformly shrinking only Swahili, Amharic, Zulu, and Khmer exposed another
dissociation:

- raw cosine and order-based retrieval readings were essentially unchanged,
  because each affected language was uniformly rescaled;
- raw direct LFS increased by `+0.226` at the median;
- shared-concept preservation fell by `-0.344` at the median;
- within-concept preservation fell by `-0.179` at the median;
- after joint standardization, AaR and mean margin worsened, but LFS moved in a
  different and model-dependent direction.

This demonstrates why mean retrieval, tail retrieval, the factor ratio, and raw
preservation must remain separate readings. They answer different questions.

## Raw LFS and dip-depth correction

The authoritative 128-language, 300-concept direct-SS grid gives dip-layer raw
LFS values from `0.644` to `0.938` across the 14 dump models. All 14 values are
above 0.5, so the language main effect exceeds the concept main effect at the
dip under this exact grid and estimator.

This corrects an assumption made while reviewing the kickoff document. The
kickoff's claimed `0.38–0.87` range does not describe these authoritative
direct-SS grid artifacts. It may refer to another estimator or panel and must
not be used without tracing its source.

The corrected table now carries:

- raw LFS at the dip;
- LFS at layer 0;
- dip depth;
- the analytic null at the exact `(L, N)`;
- excess over the null;
- an explicit language-versus-concept dominance label.

For this grid, `L=128`, `N=300`, and the analytic null is approximately
`0.2981`. Every model's dip remains substantially above that null.

The automatic text scan found 12 lines for human review. Most are quotations,
citations, or explicit warnings and should not be changed automatically. Two
clear shorthand statements require revision before publication:

- `paper/ANALYSIS_REPORT.tex:97`, which equates low LFS with a
  language-agnostic core;
- `METRIC_PROPOSAL.md:50`, which uses the same shorthand.

The authoritative technical report already contains an important warning that
LFS is not by itself evidence of language-agnosticism.

## Boundaries

- The core sweep does not validate downstream capability prediction.
- The rank intervention is a controlled representation perturbation, not a
  trained model intervention.
- Mapping-ladder `K` and the four external AQI clustering components were
  deferred until their exact implementation contracts are verified.
- The 13-language sweep and 128-language raw/depth table are separate panels.
- The two rank definitions remain separately named; no silent estimator swap
  was made.

## Next actions

1. Complete and join RMFS Profile 0.2 over all 81 scientific Aim 2
   checkpoints.
2. Run the five-layer anisotropy threat audit.
3. Run the per-coordinate language-share analysis with Beta-null unit tests,
   BY correction, and explicit Gaussian-reference wording.
4. Decide whether the trained 1.7B/4B dissociation replication is needed after
   the complete paper skeleton and compute estimate are reviewed.
