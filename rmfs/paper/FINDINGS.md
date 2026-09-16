# Findings register — the campaign as research

**Framing.** This project is a **construct-validity study of intrinsic
multilinguality metrics**: the first, to our knowledge, to (i) evaluate
metrics against multiple extrinsic criteria in a multitrait-multimethod
design, (ii) pre-register every prediction and publish every
falsification, and (iii) import the multiple-testing and
backtest-overfitting toolkit (deflation controls, cluster bootstrap,
Romano-Wolf, MCS, PBO/CSCV, effective breadth) from quantitative
finance into representation analysis. RMFS — the instrument that
emerges — is one deliverable; the validation protocol and the findings
below are the others.

Evidence classes: **[C]** confirmatory (pre-registered, frozen before
data); **[E]** exploratory (post hoc, so labeled). Effect sizes carry
95% language- or family-cluster bootstrap CIs.

---

## F0 [C] The boundary was pre-registered before it was observed.
`PREREG_BATTERY.md` (frozen 2026-07-15) records the I8 expectation that
LFS moves in the "looks better" direction under isotropic collapse,
names CVP as the detector at (1−m)², and states that effective rank
alone is an insufficient anti-Goodhart guard. `PREREG_312_313.md`
(2026-07-07) calls the collapse check "the critical one" and commits in
advance to reporting the naive scalar form as falsified if triggered.
F1/F6 are therefore confirmations of frozen predictions, not post-hoc
diagnoses — and the move to channels executed a pre-committed branch.

## F1 [C] Ratio metrics are blind to concept collapse — by construction.
A model with ~95% of concept variance removed (89% by the earlier
pipeline) posted the study's best LFS score and 3rd-best MEXA of 25 arm
families; the tail measure ranked it 12th under the current protocol
(the earlier protocol ranked it top — protocol-sensitive, recorded). In synthetic worlds, global collapse leaves
LFS-VC unchanged to 1e-6 while true concept variance falls 99%
(battery, exact closed form). Predicted in the frozen signature table.

## F2 [C] Concept-variance preservation reads collapse exactly.
C_pres = (1−m)² within 0.01 at every tested severity, uniform and
per-language (battery, 4/4 exact cells). Detection is solved;
interpretation is F6's problem.

## F3 [C] The map between languages is linear in most models, and the
exceptions are concentrated. Over 1,792 model-language pairs: linear or
simpler in 74% (linear alone 60%), nonlinear in 26%. The nonlinear
cases are not spread thinly — Falcon3-7B, OLMo-2-1B and OLMo-2-7B carry
80% of them (all three are majority-M5 models); across the other eleven
models only 6.5% of languages select a nonlinear map. Answers 3.1.1's
"just linear or more complex" with a distribution.

## F4 [C] Behavioral transfer T is health-inverted — three independent designs agree.
(i) Trained arms: collapsed checkpoints score q_T 0.19-0.33, sound
ones 0.00 (S1, 15/15 checkpoints consistent). (ii) Synthetic
dose-response: mild collapse RAISES T above baseline (+0.019 → +0.035),
monotone down through severity; total correspondence destruction nulls
it (+0.004). (iii) Cross-model: deflated ρ = −0.27 to −0.42 against
held-out content transfer. Reading: T ≈ context dependence × factor
validity; impoverished representations lean harder on injected states.
A caution for the growing family of intervention-based metrics.

## F5 [C] The structure channel predicts content transfer as well as
anything in the field. Deflated ρ(q_L, held-out content transfer) =
0.76 [0.67, 0.88] (language clusters); statistically tied with
shared-prep MEXA (paired diff +0.002, n.s.); interpretable (a variance
share, not a retrieval score). **Clustering robustness (added
2026-08-05):** the estimate is NOT an artifact of cross-language
correlation — macro-family clustering (10 clusters) gives [0.60, 0.89].
It does rest on 4 models: clustering at the model level gives
[0.12, 0.84], p=0.042 (few-cluster bootstrap, itself unreliable at
k=4). The binding uncertainty is the MODEL dimension, not the language
dimension — which is what the walk-forward adds. Reported this way
everywhere.

## F6 [C] Concept collapse at the measurement layer is behaviorally
benign (0.6B, MCQ). The 89%-collapsed arm matches its control on a
40-language downstream exam (0.395 vs 0.390; 0/3 seeds degraded;
frozen prediction P1 FAILED and binds). Massive representational
reorganization without task harm — the premise "collapse = broken
model" is false at this scale/task, and the open question "where does
the concept information go?" is F10.

## F7 [C] The damage axis in trained models is invisible to structure
metrics and partially visible to tails. Arms that actually lost
ability (−6pp) have CLEAN geometry (C_pres 0.87-1.00); AaR is the only
significant harm predictor on the fault panel (+0.51 [+0.09, +0.71]);
the structure-only composite anti-correlates (−0.53). Metric validity
measured on healthy-model panels does not transfer to fault-containing
panels.

## F8 [C] Alignment training does not improve downstream multilinguality.
Cosine word-alignment arm vs LM-only control: +0.0075 panel accuracy
(P4, bar ≤ +0.02, PASS in the null direction); the proposal's literal
MSE form degenerates representationally instead. Aim 2's premise needs
the constraint form (align subject to guards), not the naive form.

## F9 [C] Per-language validity claims rest on ~2 effective languages —
for every metric. Grinold effective breadth of the language panel:
1.2 (structure), 2.0 (MEXA) vs nominal 40-70. Cross-language
correlation makes nominal-n claims in this literature overstate their
evidence by an order of magnitude.

## F10 [E] Open: the reorganization question. A model with 5% of its
concept variance at L8 keeps full task ability (F6) and nearly full
context benefit (0.35 vs 0.38 nats). Candidate mechanisms (rerouting
through other layers; nonlinear re-encoding invisible to REML) are
testable with the layerwise profiles and patching grid.

## F11 [C] No single scalar survived honest validation; the three-channel
instrument is what the evidence supports. The soft-min composite failed
its crash test (S1 0/3) and its 4-model downstream signal was a
small-support artifact (0.24 → 0.04 at 14 models). Channels are
non-redundant: each wins exactly one exam type (F5/F7 + MEXA's 0.24-0.31
on downstream). RMFS is therefore specified as the per-language
three-channel vector + reorganization flag (A6), scalar only as a
calibrated convenience.

## F12 [C] A validation harness that kills its own claims is buildable
and cheap. Three pre-registered falsifications (LFS's blindness
premise-inversion, the scalar, the collapse-harm premise) were each
caught before publication by the same protocol: frozen predictions,
signature battery, deflation + cluster inference, fault-containing
panels. Total marginal cost over the naive pipeline: ~2 person-weeks
and ~60 GPU-hours. The protocol is model- and metric-agnostic.
