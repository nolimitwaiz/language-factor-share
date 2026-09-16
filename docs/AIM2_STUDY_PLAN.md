# Aim 2 Study: Design Document

**Status: design only. No runs launched. Written 2026-07-24.**
Author: Waiz Khan (wkhan12@jh.edu)

This document specifies a study for Aim 2 of the NSF proposal (training
methods that improve multilingual robustness). It is written to be reviewed
and cut down before anything executes. Nothing here has been run.

---

## 1. What the first study already established

A study was run under pre-registered criteria in July 2026: Qwen3-0.6B,
400 steps, objective `L = L_LM + lambda * LFS_batch`, against a
`lambda = 0` control trained on identical data for identical steps.

| Pre-registered prediction | Outcome |
|---|---|
| The quantity is optimizable (dip deepens) | **Pass.** Dip 0.206 to 0.389 in 400 steps |
| Alignment transfer beats the control | **Fail.** Mid and low tier MEXA 0.319 vs control 0.350 |
| Capability loss stays within the control's | **Fail.** Belebele -2.9 points vs control -1.7 |
| Low-resource downstream improves over base | Weak pass. 0.257 to 0.294 |

The pre-committed falsification fired. Diagnosis: the optimizer reduced the
LFS ratio by shrinking the denominator, that is by collapsing concept
representations toward each other, rather than by removing language
structure. The metric improved while the representation degraded.

A subsequent controlled injection experiment (battery cell I8b) established
how visible this failure is to each instrument. Collapsing representations
toward the global centroid at magnitude `m`:

| Instrument | Response to collapse |
|---|---|
| LFS | Improves (moves in the "good" direction) |
| MEXA | Flat even at m = 0.9 |
| AaR | Flat even at m = 0.9 |
| **CVP (concept-variance preservation)** | **Detects exactly: measured (1-m)^2 to three decimals** |

Retrieval-based measures are structurally blind here because uniform
shrinkage preserves the *ordering* of similarities, and retrieval reads
order. This is why CVP is a required component of any Aim 2 objective and
not an optional diagnostic.

**Design consequence.** Aim 2 must not minimize LFS alone. The study below
is built so that the collapse pathway is closed by construction and
monitored by an instrument proven to see it.

### 1a. A second constraint the estimator itself imposes

Measured after the study, during the July 2026 audit: under a pure-noise
null the LFS estimator returns the degrees-of-freedom ratio
`(L-1) / ((L-1) + (N-1))` exactly, because the numerator carries a noise
term `(L-1) * D * sigma^2` that does not vanish. Verified on synthetic data
across nine configurations and reproduced on real representations by a
monolingual null to three decimals.

Two consequences for this study, both of which change how an objective
should be written:

1. **LFS does not go to zero.** A model approaching true
   language-agnosticism drives LFS toward a value set by the residual
   scale, not toward zero. An objective that minimizes LFS is pushing
   against an asymptote. This is an independent reason, separate from the
   collapse finding, that the target must be constrained rather than
   minimized.
2. **The training-time batch quantity has a different null than the
   evaluation quantity.** Batch LFS is computed at small `L` and small `N`,
   where the noise term is proportionally much larger than at evaluation
   scale (128 languages, 1500 sentences). Any arm using an LFS term must
   report the batch configuration's pure-noise value alongside the loss
   curve, or the curve is uninterpretable. Arm C's original study did not
   do this, which is a reporting gap in that result rather than a flaw in
   its conclusion.

---

## 2. Arms

All arms share data, optimizer, schedule, step count, batch size, and
evaluation. Only the objective differs.

| Arm | Objective | Role |
|---|---|---|
| **A** | none (frozen pretrained) | Reference point |
| **B** | `L_LM` | Domain-adaptation control. Isolates the effect of *any* further training on this data |
| **C** | `L_LM + gamma * LFS` | Negative baseline. Expected to collapse. Included to demonstrate the failure mode under the current protocol, not to be improved |
| **D** | `L_LM + alpha * L_align` | Contrastive alignment alone |
| **E** | `L_LM + alpha * L_align + beta * L_CVP` | Contrastive alignment with the anti-collapse constraint |
| **F** | `L_LM + alpha * L_align + beta * L_CVP + gamma * LFS` | Adds the LFS term on top of E |

**Arm F exists to answer exactly one question: does an explicit LFS term
add anything beyond contrastive alignment plus variance preservation?**
If E and F are statistically indistinguishable across seeds on every
outcome, the honest conclusion is that LFS is a diagnostic and a layer
selector, not a training signal, and Arm F should not appear in any
subsequent scaling. That conclusion is an acceptable and publishable
result. The arm is powered for that comparison or it is not run.

The C-vs-B contrast is the one that reproduces the known failure; the E-vs-D
contrast tests whether CVP is doing work; the F-vs-E contrast is the
research question.

---

## 3. Loss terms

**Alignment.** Translation-equivalent sentences are positives; other
sentences in the batch are negatives. Applied to representations at a
selected layer, passed through a small projection head `g` (two-layer MLP,
output dimension 256), so that alignment pressure does not force every
hidden dimension to become language-invariant:

```
L_align = InfoNCE( g(h[c, l1]), g(h[c, l2]) ; negatives = g(h[c', ·]), c' != c )
```

with temperature a tuned constant, tuned on a development split that shares
no sentences with any evaluation set.

**Concept-variance preservation.** Measured against the frozen base model
(Arm A) on the same batch, so the reference is fixed and cannot drift:

```
CVP_batch = trace( Cov_c( h[c, ·] ) ) / trace( Cov_c( h_frozen[c, ·] ) )
L_CVP     = max( 0, tau - CVP_batch )^2
```

A one-sided hinge, so the term is inactive while concept variance is
healthy and rises sharply as it collapses. `tau = 0.90` proposed, meaning
up to ten percent variance loss is tolerated and further loss is penalized.
The exponent-2 form is chosen so the penalty gradient grows as collapse
deepens.

**LFS term.** Batch-level LFS at the selected layer, computed exactly as in
the measurement code so that the training signal and the evaluation
instrument are the same quantity.

---

## 4. Ablations

Two design choices are confounded in the arm table above and need their own
comparison, run only on the winning arm:

1. **Projection head versus direct.** Alignment applied through `g` versus
   applied directly to the residual stream. The direct variant is the more
   aggressive intervention and the more likely to damage capability; the
   head variant risks the alignment living entirely in the head and not in
   the model. Both need measuring.
2. **Which layer.** Alignment applied at the model's LFS-dip layer (read
   from the existing grid results, not recomputed) versus one-quarter depth
   versus three-quarter depth. The dip layer is the pre-registered primary;
   the others test whether the choice matters.

---

## 5. Scope

- **Model.** Qwen3-0.6B for the study, matching the first study so the
  negative result is directly comparable. Llama-3.1-8B is the natural
  scale-up candidate afterward (deepest shared space measured, 0.374, and
  exam-capable), but only after the study passes its gates.
- **Languages.** Twelve, stratified across resource tier, script, and
  family. Fixed list recorded in the run config.
- **Parallel data.** NTREX sentences disjoint from every evaluation set and
  from the sentences used in any published measurement, as in the first
  study.
- **Compute.** Single GPU, under 90 minutes per arm per seed, to remain
  backfill-friendly.
- **Seeds.** One seed while debugging. **Three seeds minimum before any
  scientific claim.** Any effect that does not survive three seeds is
  reported as not established.

---

## 6. Evaluation

Every arm is evaluated identically. Representation and preservation
measures come from the existing suite without modification.

**Representation.** Full layerwise LFS curve; language variance; concept
variance; residual share; CVP against the frozen model; MEXA; AaR at the
{1, 5, 10, 20} percentiles.

**Behavior.** Cross-lingual context use (the generation-time measure from
the confirmation experiments); Belebele 0-shot overall and by resource
tier; requested-language generation.

**Preservation.** LM loss and perplexity on held-out text; English-only
capability; monolingual capability in two non-English languages;
representation effective rank; covariance spectrum; concept retrieval.

Note on what "collapse indicators" means concretely here: effective rank
and CVP together, because the battery established that effective rank alone
is blind to isotropic collapse (it is scale-invariant) while CVP catches it
exactly.

---

## 7. Success criteria, declared before running

**A falling LFS is not success.** An arm is promising only if **all** of
the following hold, jointly, in three of three seeds:

1. LFS falls at the targeted layers relative to Arm B (not merely relative
   to Arm A, which would confound the objective with ordinary continued
   training).
2. `CVP >= 0.90` against the frozen model throughout training.
3. Residual variance share does not rise by more than 5 points.
4. MEXA or AaR improves relative to Arm B, or holds within noise.
5. Cross-lingual context use improves relative to Arm B.
6. English capability and monolingual capability each degrade by no more
   than Arm B degrades, plus 1 point.
7. The direction of every effect above is consistent across all three
   seeds.

Failing any one of these is a failure of that arm and is reported as such.
Arm C is expected to fail criteria 2 through 6 and is included precisely to
show the contrast.

---

## 8. What would make this study uninformative

Stated in advance so it is recognizable when it happens:

- If Arm B (ordinary continued training) already moves LFS substantially,
  the metric is responding to data domain rather than to the objective, and
  every comparison must be made against B rather than A.
- If the alignment loss saturates within a few dozen steps, the batch or
  temperature is wrong and the arms are not being compared at matched
  effective pressure.
- If all arms including C preserve CVP, the collapse pathway is not
  reachable at this scale and step count, and the study cannot speak to the
  question it was built for.

---

## 9. Open questions for co-design

These are the choices I would rather make with Professor Koehn and
Professor Murray than alone:

1. Whether alignment pressure belongs at the dip layer (where our
   measurements say the shared space is deepest) or earlier (where the
   proposal's word-alignment framing might place it).
2. Whether the parallel-data arm should be joined by a code-switched arm,
   as the proposal's Aim 2 describes, and whether that is a separate study
   or a fourth condition here.
3. Whether the target model for scale-up is Llama-3.1-8B (deepest measured
   shared space) or an open-data model such as OLMo-2, where the training
   mixture is known and the result would be more interpretable.
4. Whether preference-style training, which the proposal also mentions,
   is in scope for the first study or deferred.
