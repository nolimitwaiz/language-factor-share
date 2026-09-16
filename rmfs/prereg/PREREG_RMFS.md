# PREREG_RMFS.md — Pre-registration for the RMFS metric (v1)

STATUS: **FROZEN** at tag `prereg-rmfs-v1`. Changes only via dated addenda
in `prereg/addenda/`, each written before the run it affects.

Sign-off: Waiz Khan (freeze authorized in session)  Date: 2026-08-02  Commit: tag prereg-rmfs-v1

---

## 1. Object under test

RMFS_{ℓ,k}: a per-language multilingual robustness score at layer k,
aggregating four components by soft minimum. Claim class it will be tested
for: (a) descriptive fault detection (synthetic battery, intervention
arms), (b) per-language deflated predictive validity, (c) generalization to
unseen model families, (d) non-reversal across the existing intervention
arms. It is explicitly NOT claimed to be a safe optimization target until
the Phase-Next constrained-training arm is run and scored.

## 2. Fixed protocol

- Text: NTREX-128, first 300 sentences per language (development protocol);
  campaign dumps (n=1500) used where they exist. Document IDs carried
  through all splits.
- Downstream: Belebele 0-shot log-likelihood, 300 questions/language.
- Layer rule (frozen): the LFS-VC dip layer per model, computed once on the
  development protocol; sensitivity report at fixed relative depth 0.5.
- Preprocessing: fp32 pooling; per-dimension z-scoring in the native track.
  A geometry-invariant track (centering + global scalar normalization only)
  is computed for the stability requirement S9. Any fitted transform is fit
  on training concepts only.
- Splits: document-purged, concept-disjoint, 20 folds, seeds {13, 42, 71}.
- Covariates for deflation: log10 CulturaX tokens (CC-100 backstop,
  imputation flagged), Glottolog-style macro-family, script, tokenizer
  fertility. Estimator bracket: raw-residual, family-demeaned, median-s²
  disattenuation; language-cluster bootstrap (2,000 reps) for CIs.
- Ladder parity (Phase-1 acceptance): the new-stack ladder, run with the
  LEGACY (unpurged) splits for the parity check only, must reproduce the
  legacy per-rung grid-mean shares within **1.0 percentage point per
  rung**. Purged-split deltas are findings, reported not gated (the Phase
  0.3 audit quantifies them first).

## 3. Component definitions (frozen)

### 3.1 T — behavioral functional transfer
Via the 3.1.3 harness. For language ℓ at layer k:
T_ℓ = (B_recon − B_mismatch) / (B_native − B_mismatch), where B = target
NLL reduction versus no-context, B_recon uses context token states
reconstructed from the shared factor through ℓ's selected ladder map (fit
on training concepts, applied to held-out concepts, document-purged).
Raw T is reported unclipped. Normalized: q_T = clip(T, 0, 1).
Degenerate-denominator rule: if B_native − B_mismatch < ε_B = **0.072
nats**, T for that language is reported as undefined, not zero. Provenance:
pooled 5th percentile of the 'content' field (delta_matched −
delta_mismatched) over the four legacy §3.1.3 models × 40 languages
(160 cells, all positive; per-model p5: OLMo-2-1B 0.041, Qwen3-0.6B 0.146,
Qwen3-1.7B 0.206, Qwen3-4B 0.204; source
`legacy` results/transfer313/*_v2.json, computed 2026-08-01 before freeze).

### 3.2 LFS-VC — language share of systematic variance
REML variance components of h = μ + a_c + b_ℓ + e.
LFS-VC = σ̂_L²/(σ̂_L² + σ̂_C²). Normalized component: q_L = 1 − LFS-VC.
Negative raw component estimates are flagged; REML is primary.

### 3.3 K — mapping complexity
M0 identity, M1 offset, M2 offset+scale, M3 orthogonal, M4 linear,
M5 nonlinear (kernel). Selection: smallest rung whose document-purged
held-out reconstruction loss is within one SE of the best rung.
K_ℓ = rung/5. Normalized: q_K = 1 − K_ℓ.

### 3.4 C_pres — concept-variance preservation (intervention mode only)
C_pres = min(V_concept/V_concept,ref, 1), ref = the frozen base model of
the intervention. Normalized: q_C = C_pres.
Observational mode (no ref exists): RMFS-obs aggregates [q_T, q_L, q_K]
only; capacity diagnostics (r_eff, norm, probe-MDL) are reported as
pass/fail gates beside it. Thresholds are set by **pre-named addendum A1**
(`prereg/addenda/A1_capacity_gates.md`), computed from the distribution of
the 17 pretrained models and written before any arm is scored; scoring any
arm in observational mode before A1 exists is a protocol violation. The
numbers are deferred, the procedure is not: A1 sets r_eff at the 5th
percentile and probe selectivity at the 5th percentile of the 17-model
distribution at the frozen layer rule.

### 3.5 Aggregation
RMFS = −τ · log( (1/J) Σ_j exp(−q_j/τ) ), τ = 0.1 (frozen).
J = 4 intervention mode, 3 observational mode. The component vector is
always published beside the scalar; the scalar is a ranking statistic only.

## 4. Signature matrix (frozen expectations; numeric tolerances in
`prereg/signatures/rmfs_signatures.csv` — 75 cells, written 2026-08-01
before freeze. Conventions: `band` |Δ| ≤ k·floor; `down`/`up` beyond
k·floor in the named direction; `down_mag` additionally monotone in the
injected magnitude; `ceiling` no rise beyond k·floor, any fall allowed;
`exact` closed form within abs_tol (uniform collapse C_pres = (1−m)²
± 0.01, calibrated in prior battery cell I8b); `rung_le`/`rung_ge`
discrete ladder-rung expectations. k = 3; `floor` = the per-component
identity-injection noise floor, measured at battery calibration BEFORE any
scoring)

Injection (calibrated magnitudes)  | T      | 1−LFS-VC | 1−K   | C_pres | RMFS
-----------------------------------|--------|----------|-------|--------|------
Identity                           | ≈      | ≈        | ≈     | ≈      | ≈
Additive offset (per language)     | ≈      | ↓        | ≈/↓*  | ≈      | ↓ mild
Isotropic scale (per language)     | ≈      | ↓        | ≈/↓*  | ≈      | ↓ mild
Global shared rotation             | ≈**    | track-dep| ≈     | ≈      | ≈ (invariant track)
Per-language rotation θ∈{.1,.4,.8} | ↓ w/ θ | ≈/↑res   | ↓↓    | ≈      | ↓
Shear / general linear             | ↓      | ≈/↑res   | ↓↓    | ≈      | ↓
Low-rank C×L interaction           | ↓      | ≈        | ↓     | ≈      | ↓
Monotone warp                      | ↓      | ≈        | ↓↓    | ≈      | ↓
Uniform collapse m∈{.9,.5,.1}      | ↓↓ w/ m| ≈ (blind)| ≈     | ↓↓=m²  | ↓↓
Per-language collapse              | ↓ (ℓ)  | ↑        | ≈     | ↓ (ℓ)  | ↓
Pairing permutation                | ↓↓     | ↑        | ↓     | ≈      | ↓↓

\* offset/scale are absorbed by rungs ≤ M2, so K stays low — this is the
   design: cheap misalignment is not severe failure.
\** T uses a refit map per condition; a shared rotation with refit map must
   leave T within its identity noise band — this is stability, not blindness.
"≈" means within the identity-injection noise floor measured in calibration.
Every cell is scored; any miss is reported as a battery failure.

## 5. Survival requirements (all scored; failures reported; no
retrospective redesign before the full battery completes)

S1. Rank WA-C (collapsed word-MSE arm) below WA-G (cosine, sound) and the
    LM-only control on RMFS, driven by q_T and/or q_C, on ≥ 2 of 3 seeds
    per comparison.
S2. Detect uniform collapse (all three m levels) and per-language collapse
    per the signature matrix.
S3. Stability: shared orthogonal rotation and function-preserving global
    rescaling leave RMFS (invariant track; refit maps) within the identity
    noise band.
S4. Respond to injected offsets, per-language rotations, interactions, and
    warps per the signature matrix, with calibrated magnitude recovery
    where specified.
S5. Predict held-out cross-lingual content transfer after deflation:
    pooled residual Spearman of RMFS (obs mode) with R_content ≥ 0.30,
    p < 0.05 under language-cluster bootstrap, on the 3.1.3 model set.
    (Note: T shares machinery with R_content; the S5 headline therefore
    also reports the T-excluded aggregate [q_L, q_K] against R_content as
    the circularity-free line.)
S6. Per-language signal beyond observables: deflated |ρ| of RMFS with
    transfer alpha, lower CI bound > 0 under the primary (raw-residual)
    scheme on the exam-capable subset.
S7. Increment beyond MEXA: orthogonalized-RMFS deflated correlation with
    alpha, lower CI bound > 0 — OR a demonstrated unique fault-detection
    role (S1/S2 pass where MEXA is flat) if the increment fails. Which
    claim is made depends on which passes; both outcomes are reported.
S8. Generalization: on the two pre-named unseen families (see §6), deflated
    validity within or above the development-grid CI, and battery-relevant
    fingerprints scored blind.
S9. Protocol validity: cross-model RMFS ordering rank-correlation ≥ 0.8
    within each track (native protocols vs native; invariant vs invariant)
    across the pooling variants named in §7. Cross-track agreement is
    reported but NOT a pass/fail criterion.
S10 (rescoped). Non-reversal on existing interventions: across the 28 arms
    x 3 seeds, the Spearman association between RMFS and held-out
    downstream performance is non-negative, and in no objective family does
    it reverse sign significantly (bootstrap over seeds). The full
    optimize-RMFS training test is Phase-Next, pre-registered separately
    before it runs.

## 6. Walk-forward families (must be untouched by Phases 0–3)

Pre-named and FINAL: family A = **Gemma-3-4B-pt**; family B =
**Aya-Expanse-8B** (deliberately multilingual, the class the grid lacks).
Pre-named fallback if gating/licensing blocks either: **EXAONE-3.5-7.8B**.
Exact HF ids are resolved at Phase-4 download time (metadata resolution is
not "touching" the family; any extraction or analysis before Phase 4 is).
After freeze the names may not be swapped except via the pre-named
fallback, with a ledger entry.

## 7. Sensitivity set (reported, not selected on)

Pooling: mean (primary), final-token, position-weighted; layer: dip rule
(primary), fixed 0.5 depth; n: 300 vs 1500 where dumps exist; estimator
bracket as §2. The primary configuration is the one named here; all others
are sensitivity columns.

## 8. Multiple testing and selection accounting

Romano–Wolf stepdown over the S5–S7 comparison family; Model Confidence Set
(90%) over the tournament candidates; PBO via CSCV over the frozen
configuration set {layer rule, τ, clipping, pooling primary}. Every degree
of freedom fixed in this document is listed in the CSCV configuration
manifest so the selection cost is measured, not assumed away.

## 9. Reporting commitments

The component vector is published wherever the scalar is; every survival
requirement's verdict appears in the paper (main text or appendix) whether
pass or fail; all battery misses are itemized; the ERROR_LEDGER is
summarized in the appendix. If RMFS fails S6 and S7's increment branch, the
paper's claim downgrades to the audit-paper framing with RMFS as a
fault-detection instrument — this downgrade path is accepted now, in
advance.
