# PRE-REGISTRATION: Synthetic Battery + Misalignment Budget (Round 1)
# STATUS: FROZEN 2026-07-15 (user authorized freeze-with-recommended-defaults
# in session; all [DECIDE] fields resolved below, resolutions marked
# [FROZEN:]). No battery or budget run has executed at freeze time (Phase-0
# dumps + parity only). Amendments hereafter only as dated addenda added
# BEFORE the affected run. Code emits numbers; pass/fail judgment happens
# against this document only.
#
# FREEZE-TIME SIGNATURE CORRECTIONS (2026-07-15, pre-run, part of freeze):
# (a) I4 x LFS: draft said "~0". Corrected to "+ (rises with theta)": rotated
#     languages decorrelate cross-language concept means -> concept share
#     falls -> the LFS ratio RISES. LFS responds to rotation damage but
#     CANNOT ATTRIBUTE it (indistinguishable from generic shared-structure
#     loss); attribution is the budget's job. This sharpens, not weakens,
#     the construct-boundary claim.
# (b) I8 x effective rank: draft said "--". Corrected to "~0": ISOTROPIC
#     collapse scales all singular values equally and effective rank is
#     scale-invariant. Detector for isotropic collapse = CONCEPT-VARIANCE
#     PRESERVATION (CVP, added as an instrument, predicted --). Effective
#     rank catches only anisotropic collapse. Pinned consequence: effective
#     rank ALONE is an insufficient anti-Goodhart guard; G4 uses CVP + rank.

Drafted: 2026-07-15 (agent skeleton). Frozen: 2026-07-15.

## 0. Scope

Instruments under test: incumbent LFS (unchanged), MEXA, AaR@10, the nested
mapping-ladder budget (M0–M5 shares), alignment-sensitivity gap (DIAGNOSTIC
ONLY — never reported as "rotation share"), linear language-probe accuracy,
kernel-vs-linear CKA gap, denoised effective rank. Injection point: pooled
PRE-z-score fp32 embeddings; the full standard pipeline (z-scoring included)
runs untouched downstream — the battery tests the pipeline end to end.

Prototype models (Phase 0 dumps): Qwen3-1.7B-Base (fp16), OLMo-2-0425-1B
(fp16), bloom-1b7 (fp32). Layers per model: L0, ~0.25 depth, LFS-dip, ~0.75,
final (dip read from prior grid results, never recomputed).

## 1. Parity gate (blocks everything)

LFS and MEXA recomputed from Phase-0 dumps through the standard pipeline must
match the prior grid numbers.
- Tolerance: d_LFS <= 0.005; per-language mean d_MEXA <= 0.01.
  [FROZEN: gate already executed and PASSED 2026-07-15 on c24/c25 (grid
  hardware): worst d_LFS 6e-7, worst mean d_MEXA 2e-4, all 3 models,
  5 layers each, 127 languages.]
- HARDWARE CAVEAT [CONFIRM]: the original grid computed hidden states on
  cluster GPUs (2080Ti fp16; bloom fp32); Phase-0 dumps are computed on Apple
  MPS. Kernel/accumulation-order differences can move hidden states beyond
  quantization noise. If parity fails within ~2x tolerance AND the fp16
  round-trip quantization check is clean, the pre-registered interpretation
  is "hardware delta, not pipeline bug" — resolution: rerun ONE model's dump
  on c24/c25 (grid hardware) and require parity there. If cluster parity also
  fails, it IS a pipeline bug and everything stops.

## 2. Injection set and magnitude grids

Applied to non-pivot languages except where noted; English pivot untouched
(the pivot's role in MEXA/AaR makes pivot-injections a separate experiment).
[FROZEN: two tiers of cells —
 PANEL cells: the curated 26-language tier panel (TIER dict in
 cluster_grid.py, intersected with dump languages) + eng pivot; full
 instrument set including the ladder budget; used for all magnitude/theta
 grids.
 FULL cells: all 127 non-pivot languages, ONE magnitude per injection
 (marked * below); pipeline metrics only (LFS/MEXA/AaR/CVP/eff-rank), no
 ladder (compute).]

- I1 additive offset, m in {0.25, 0.5*, 1.0} x RMS.
- I2 isotropic scale, log-uniform in ±log(1+m), m in {0.25, 0.5*, 1.0}.
- I3 shared GLOBAL rotation (full-D, same Q for all languages incl. pivot)*.
- I4 language-specific rotation in top-r subspace,
  theta in {0.1, 0.2, 0.4*, 0.8} rad, r = 32 (see §3).
- I5 shear (anisotropic diag scale in subspace), m in {0.25, 0.5*}.
- I6 concept-dependent low-rank interaction, rank k = 4,
  strength m in {0.25, 0.5*}.
- I7 monotone nonlinear warp (per-language mixed tanh squash),
  m in {0.5, 1.0*}.
- I8 partial collapse toward language centroid, m in {0.3, 0.6*, 0.9}.
- I9 pairing permutation (full shuffle)*.
- I10 identity, run TWICE on independently resampled sentence halves
  (noise floor; doubles as split-half reliability samples)*.

Layers: each model's LFS-dip layer (primary); I3 and I10 additionally at L0
and the final layer (pipeline-invariance checks). [FROZEN]

Seeds: one deterministic master seed per (model, layer, injection, magnitude,
rep) cell via crc32 of the cell name, recorded in the ledger. One seed per
cell (grids provide the replication axis); I10 runs twice by construction.
[FROZEN]

## 3. Subspace and ladder settings

- Rank grid at n=300: {16, 32, 64}; MP edge = ceiling; selection by held-out
  M4 reconstruction error, one-SE rule. Never on downstream numbers.
  [FROZEN: rank selection runs once per model on the UNINJECTED real-data
  budget; battery cells use fixed r = 32 (the grid midpoint) so that cell
  responses are comparable across injections. If real-data selection lands
  at 16 or 64 for any model, the real-data budget uses the selected rank and
  the battery's r=32 stands as-is — cells are internally controlled by the
  I10 floor at the same rank.]
- Concept split 50/50, stratified by sentence-length tercile [FROZEN];
  S = 20 splits for the real-data budget, S = 5 for battery cells [FROZEN];
  all shares reported as mean ± SD across splits, TEST halves only.
- GPA consensus on TRAIN concepts; gauge anchored to the raw cross-language
  mean configuration (pinned in gpa.py — anchoring makes M0–M2 entries
  well-defined; rungs >= M3 are gauge-invariant regardless).
- Permutation null: B = 200 (M3, M4; M5 rung skipped in these refits),
  B = 50 for M5 (kernel refits are the bottleneck; null shared across
  splits), B = 50 (M1, M2 if reported). Nulls run on the REAL-DATA budget
  only; battery cells are controlled by I9 (full permutation cell) and the
  I10 noise floor. [FROZEN]
- A rung's share is claimed nonzero only if it exceeds the 95th percentile
  of its own null. [FROZEN]

## 4. Frozen signature matrix (predictions)

Magnitude classes: 0 (within I10 noise floor), + (small), ++ (large),
– (decreases). Cells marked * encode mechanical couplings DISCOVERED DURING
TEST CONSTRUCTION (2026-07-15, before any real-data battery run — hence
admissible as frozen predictions):

| Injection | LFS | MEXA | AaR@10 | budget: offset | scale | rotation | linear | nonlinear | probe acc | CKA gap | eff. rank | CVP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 offset | ++ | – | – | ++ | 0 | 0 | 0 | 0 | ++ | 0 | ~0 | ~1 |
| I2 scale | + | – | – | 0/+ | ++ | 0 | 0 | 0 | + | 0 | ~0 | ±(=s²) |
| I3 global rot | 0† | 0† | 0† | 0† | 0† | 0† | 0† | 0† | 0† | 0† | 0† | ~1† |
| I4 lang rot | +‡ (rises with theta) | –– | –– | +* | +* | ++ (rises with theta) | 0 | 0 | + | 0 | ~0 | ~1 |
| I5 shear | + | – | – | + | +* | +* | ++ | 0 | + | 0 | –/~0 | ± |
| I6 interaction | ~0/+ | – | –– | 0 | 0 | +* | ++ | 0/+ | + | 0/+ | ~0 | ~1 |
| I7 warp | ~0/+ | – | – | 0 | 0 | + | +* | ++ | + | ++ | –/~0 | –(mild) |
| I8 collapse | – (looks "better") | – | – | 0 | ++* | 0 | 0 | 0 | – | 0 | ~0‡ | –– (=(1–m)²) |
| I9 permutation | ~0 (main effects unchanged) | ≈0 floor | –– | all shares ≈ null | | | | | ~0 | 0 | ~0 | ~1 |
| I10 identity x2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ~1 |

‡ = corrected at freeze (see header): I4/LFS rises via concept-share loss
(response without attribution); I8/eff-rank flat because isotropic collapse
is scale-invariant — CVP is the collapse detector.

† I3, stated carefully (pre-registered as a finding either way):
pre-normalization quantities are EXACTLY invariant; the post-z-scoring
pipeline is only APPROXIMATELY invariant because per-dimension z-scoring is
not rotation-invariant. Prediction: all deltas flat WITHIN THE I10 NOISE
FLOOR; the measured deviation is reported as the anisotropy guard's gauge
sensitivity.

Mechanical couplings pinned in advance (the * cells):
1. I4: rotation about the POOLED mean displaces each language's centroid ->
   small mechanical offset (and scale) shares are EXPECTED; prediction is
   "rotation rung dominant and rising with theta", not "offset = 0".
2. Pure-rotation share splits: an isotropic shrink reduces error for ANY
   rotated copy (optimal s = mean diagonal cosine), so M2 legitimately
   absorbs part of a rotation. Prediction: M3 largest AND E3 ~ 0.
3. General-linear maps: similarity Procrustes explains most of a
   stretch-between-rotations map. Prediction for I5/I6: "M4 COMPLETES the
   fit (E4 ~ 0, E4 << E3) and clears its null", not "M4 majority share".
4. I8 collapse appears in the budget as a SCALE effect (shrink toward
   centroid ~ isotropic scale in the consensus map) — the detector that
   separates collapse from benign scale is EFFECTIVE RANK (––) plus probe
   accuracy; this is the anti-Goodhart tripwire (G4).

Calibration curves (required for G1):
- I4: recovered rotation angle (Procrustes on injection-basis coords,
  relative-rotation spectrum) vs injected theta — monotone, mean recovered
  angle within ±20% of injected theta. [FROZEN]
- I2: recovered scale s_hat (similarity-Procrustes) vs injected s — within
  ±10%. [FROZEN]

## 5. THE TEETH — real-data validity predictions (prototype models, then campaign)

At each model's dip layer, per-language rotation share and M4 (linear) share
from the ladder -> partial Spearman vs transfer alpha, controlling the
standard observables (log tokens, macro-family, script, fertility) AND
per-language alignment at the LFS-selected layer.
- Pre-registered sign: NEGATIVE (rotation is a failure channel LFS misses).
- n for the prototype run: 3 models x ~120 languages (pooled with model FE).
- Either outcome publishes: a null = "the additive boundary is empirically
  harmless on this grid" — also a paper result.
- p < 0.05, BH-corrected over the {rotation share, M4 share} x {dip layer}
  family (2 tests; best-layer variants are exploratory appendix). [FROZEN]

## 6. Campaign launch conditions (STOP-3 gates, all required)

- Battery scorecard reviewed against this frozen matrix;
- calibration curves monotone within tolerance;
- rank selection stable across the 3 prototype models (selected r identical
  or adjacent on the grid);
- this file FROZEN with all [DECIDE] resolved;
- explicit user go-ahead.
Campaign: n = 1500 sents/lang [FROZEN]; models = 10 grid + SmolLM2 +
Falcon3 + Salamandra-2B/7B (gating to be verified on the login node before
launch; backup Teuken-7B) [FROZEN]. G5 governance holdout = the new model
(SmolLM2/Falcon3 are spent from the walk-forward).

## 8. Section 9 (checkpoint trajectories): OPTED OUT at freeze.
No user opt-in was given; the module is not built in Round 1. Re-opening
requires a dated addendum before any trajectory run.

## 7. Governance gates (incumbent LFS vs challengers)

G1 battery signature + calibration pass; G2 held-out deflated behavioral
validity improvement >= [DECIDE] under language-cluster bootstrap; G3 share
stability across splits and adjacent ranks (SD <= [DECIDE]); G4 anti-collapse
(no reward on I8 rows); G5 generalization on the untouched family.
Otherwise LFS stands and the budget publishes as diagnostic instrumentation.
Either outcome is a paper result.

---

## ADDENDUM A1 (dated 2026-07-16, added BEFORE any A1-affected run)

Round-1 battery revealed two coverage gaps; two cells are added. No prior
cell, prediction, or gate is modified. Predictions below are frozen as of
this date; the A1 patch run executes only after this addendum is committed.

### I4b — centroid-preserving language rotation
Motivation: I4 rotates about the POOLED mean, so for offset-dominated models
(OLMo) most injected energy becomes centroid displacement and the rotation
rung shows no power. I4b rotates each injected language's cloud about its
OWN centroid in the top-r subspace: x' = c_l + R(x - c_l). Grid
theta in {0.1, 0.2, 0.4, 0.8}; panel cells with budget; full cell at 0.4.
Frozen predictions:
- Language centroids unchanged EXACTLY -> budget offset rung ~unchanged;
  scale rung ~unchanged.
- Rotation rung ++ rising with theta FOR ALL THREE MODELS — this is the
  power-restoration claim; if OLMo's rung still fails to respond, the
  weak-power finding is real and model-geometry-intrinsic, not an artifact
  of the I4 design.
- LFS + (concept-share loss, responds-without-attributing), MEXA/AaR
  degrade with theta, CVP ~1, eff-rank ~0, recovered theta exact (centered
  Procrustes recovery).

### I8b — collapse toward the GLOBAL centroid
Motivation: I8 collapses toward each language's own centroid, which RAISES
LFS. The Aim-2 pilot's gaming mode must therefore have been collapse toward
the global/shared centroid; the battery currently has no such cell.
I8b: x' = mu_global + (1-m)(x - mu_global) for non-pivot languages,
m in {0.3, 0.6, 0.9}; panel cells with budget; full cell at 0.6.
Frozen predictions:
- LFS DECREASES (looks better) with m — the Goodhart direction, reproducing
  the pilot's gaming signature inside the battery.
- MEXA/AaR degrade (–– at high m); CVP = (1-m)^2 exactly; eff-rank ~0
  (isotropic); probe 0/– at high m (languages crowd together).
- THE TRIPWIRE PATTERN (the point of the cell): LFS improving while CVP
  collapses and MEXA degrades = the suite's anti-Goodhart contradiction
  signature. G4 evaluation for any Aim-2 objective uses I8b rows alongside
  I8.

---

## ADDENDUM A2 (dated 2026-07-17, added BEFORE any A2-affected run)

### Campaign parameters (STOP-3 go-ahead given in session; rank-stability
### gate WAIVED with mitigation)
- n = 1500 sentences/language, all NTREX languages.
- Models: the 10 grid models + SmolLM2-1.7B + Falcon3-7B + Salamandra-2B +
  Salamandra-7B (G5 governance holdouts; both confirmed ungated;
  backup Teuken-7B).
- Saved artifacts: pooled pre-z fp32-pooled embeddings (fp16 storage) at 5
  selected layers; per-sentence retrieval margins per (language, layer) —
  enables AaR@{1,5,10,20}.
- Rank-stability mitigation (gate failed 16/16/64 at n=300): all campaign
  budgets reported at BOTH the per-model selected rank and common r=64.
- Subspace variant: budgets additionally computed with the subspace defined
  on LANGUAGE-CENTERED pooled data (removes offset directions from the
  basis; motivated by A1's OLMo power finding). Predictions: for
  offset-dominated models the centered-subspace rotation rung becomes
  responsive to injected rotation (power restored); real-data rotation
  shares remain < 1% for all models (the harmless-boundary claim, now
  testable with power everywhere).
- Whitened-pipeline robustness variant (motivated by I3): LFS/MEXA
  recomputed after per-layer ZCA whitening. Prediction: cross-model RANKING
  of dip depths is preserved (Spearman > 0.8 vs native-basis ranking);
  absolute values may shift.
- Kernel rung: prediction — at n=1500 (750 train concepts) the M5 negative-
  share overfit pathology disappears (mean M5 share in [-0.02, +0.05] for
  all models).

### Salamandra fingerprint predictions (FROZEN before its first run)
Salamandra (BSC): modern, deliberately balanced 35-language European
training mix — the "trained-for-coverage done right" control that separates
BLOOM's coverage-without-convergence from undertraining.
- P-S1: U-shaped LFS fingerprint (not BLOOM-flat); dip depth in the
  intermediate-to-deep class: 0.10-0.30 at 2B.
- P-S2: dip deepens (or holds) from 2B to 7B — NOT the BLOOM invariance.
- P-S3: European-language MEXA high / non-European low at the dip (EuroLLM-
  like tier signature), gap larger than Qwen3's.
- P-S4: hub/donor structure: English NOT the best donor for most European
  targets (intra-European donors dominate), consistent with the
  latent-factor finding.
Scoring: P-S1/P-S3 at 2B grid run; P-S2 needs both sizes; P-S4 at dip layer.
Failures reported as always.

---

## ADDENDUM A3 (dated 2026-07-19, added BEFORE any A3-affected run)

### Two additional candidate exam-capable families (hardening item:
### widen the deflated-validity model base beyond 4 capable families)
Salamandra failed the frozen exam-capable criterion (2b: 0.250; 7b: 0.2945
< 0.30 aggregate — excluded per rule, near-miss disclosed). Replacement
candidates, both ungated, both new labs to the grid:
- ibm-granite/granite-3.1-8b-base (granite: the pre-named fallback family
  from WALKFORWARD_PROTOCOL)
- 01-ai/Yi-1.5-9B

Frozen predictions (scored on grid + 0-shot Belebele, same protocols):
- P-A3.1 Both models are exam-capable (aggregate 0-shot Belebele >= 0.30).
- P-A3.2 Both show U-shaped LFS fingerprints in the intermediate-to-deep
  class (dip depth 0.10-0.35); neither is BLOOM-flat.
- P-A3.3 Yi-1.5 (zh/en-heavy recipe) patterns with the Qwen class:
  dip depth > granite's.
- P-A3.4 Adding whichever models qualify to the deflation refresh does NOT
  collapse the exam-capable validity bracket: deflated alpha-correlation
  on the widened capable subset stays >= 0.45 (current: ~0.65).
Failures reported as always; exam-capability rule applied as frozen.

### A3 extension (dated 2026-07-19, BEFORE the Llama run; access granted)
Third added family: meta-llama/Llama-3.1-8B (gate accepted by waizkhan).
- P-A3.5 Exam-capable (aggregate 0-shot Belebele >= 0.30), comfortably.
- P-A3.6 U-shaped fingerprint, intermediate-to-deep dip (0.15-0.35);
  EN-centric-leaning tier signature (closer to Mistral/Falcon than Qwen3).
