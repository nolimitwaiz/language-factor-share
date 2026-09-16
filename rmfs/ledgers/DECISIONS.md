# DECISIONS

## Phase 0 report (2026-08-01)

**Built.** Repo at ~/dev/rmfs (off iCloud; Desktop confirmed synced) +
cluster mirror ~/rmfs. Scaffold per CLAUDE.md; pyproject/ruff/pytest;
configs generated from committed prior-campaign results (17 models, dip
layers, precision rules); NTREX symlinked with document IDs; legacy copied
byte-identical and read-only (14 modules, 8 tests, 6 preregs, sha256 in
INVENTORY); MEMORY.md (history, numbers, D1-D10, artifact map); proposal
objectives extracted from the 24pp PDF into CLAUDE.md; prereg finalized
with zero TODOs and a 75-cell numeric signature matrix.

**Passed.**
- Inventory complete with 7 verified corrections to the kickoff pack
  (stale paths, study_metrics naming, 28=25+3 arms, NO arm dumps exist,
  shared volume ~97% full, splits unpurged, local dumps empty).
- Purged-split audit: leakage quantified (99.3% / 99.9% of held-out
  sentences share a document with train at n=300 / n=1500); purged
  re-estimate on Qwen3-1.7B with EXACT parity to the published table
  (max error 2e-14) and one flag: M5's overfitting was understated by
  ~6pp under leaky splits. Substantive conclusions (additive dominance,
  negligible rotation) survive within ~1pp.
- import rmfs + ruff clean; 3 commits, sole author.

**Failed (all productive, all ledgered).**
- Job 1703653: legacy geometry modules are NOT standalone (relative
  imports) — INVENTORY corrected, package shim added.
- Job 1703711: DOCUMENT_IDS.tsv never existed on the cluster; the legacy
  3.1.3 loader silently degrades without it (E2) — every cluster-side
  3.1.3-family run lacked the document filter. Hard-fail rule adopted.
- Job 1703720: wall-time estimate wrong by ~7x (1 split ≈ 30 min serial);
  cancelled, fitting parallelized with sequential index generation
  preserving rng fidelity. Completed in 1:51:45.

**Open questions / carried items.**
- Prior-campaign erratum queued: the published M5 row understates
  overfitting by ~6pp (E1 resolution).
- Phase 2 needs arm embedding extractions from checkpoints (no dumps
  exist); size estimate required against the 97%-full shared volume.
- HF token rotation still pending (user action, carried from prior repo).

**Gate.** Phase 0 acceptance is met except its final item: the prereg
freeze. Awaiting Waiz's signature + git tag prereg-rmfs-v1. Nothing in
Phases 1-4 runs before the tag; no RMFS number exists anywhere.

## Phase 1 report (2026-08-02)

**Built.** Measurement core complete, tests-first throughout: REML variance
components + LDE (variance.py), gates (C_pres exact s^2 law, effective
rank, norm, hard-fail dPPL hook), Ledoit-Wolf shrinkage, the M0-M5 ladder
port with document-purged cross-fitting and one-SE selection, tournament
baselines (shared-prep MEXA/AaR at 1e-12 legacy parity, CKA, probes with
MDL+selectivity, official-pooling bridge kept separate), the deflation
port with the full estimator bracket and cluster-bootstrap CIs.

**Acceptance - all four items met.**
1. Unit battery green: 51 tests on cluster python 3.11.9 (the enforced
   authoritative environment via the sbatch prologue).
2. Ladder parity with legacy: EXACT (1e-12 shares, 1e-10 consensus).
3. Deflation bracket reproduced: all four published MEXA/AaR numbers to
   four decimals (+0.0000), R2 matching, tokens gate PASS.
4. LDE and q_K deflated-validity numbers with CIs: exist, manifested
   (results/runs/component_brackets).

**The scientific result of the phase: both geometric components deflate to
null.** LDE raw -0.336 -> deflated +0.025 CI[-0.05,+0.09]; q_K raw +0.094
-> deflated -0.009 CI[-0.07,+0.05]; family-demeaned and disattenuated
schemes agree. Neither deviation energy nor mapping complexity predicts
downstream transfer beyond tokens/family/script/fertility on this grid.
Consequence, stated before T exists: RMFS's predictive-validity case
(S5/S6, S7's increment branch) now rests entirely on the behavioral
component. The fault-detection case (S1/S2, S7's second branch) is
unaffected - it never claimed deflated validity. The prereg's accepted
downgrade path exists for exactly this contingency.

**Findings en route.** First REML LFS-VC grid (0.59-0.90; construct stable
vs prior stack within 0.003-0.064 across the protocol change). N5:
raw-track effective rank measures massive-activation anisotropy (12/14
models ~1; OLMo-2 24-78); A1 must specify the track. Rung distributions:
M4 dominates; easy languages select cheap maps; OLMo-2 selects M5 for all
128 languages and Falcon3 for 120 - flagged, uninterpreted.

**Failed and ledgered.** pytest missing from the cluster venv (prologue
refused, as designed); OLMo-2-1B budget primary never existed (explicit
recorded fallback added); E3 BLAS oversubscription (fork inherits thread
state; OMP_NUM_THREADS=1 must be exported before python starts) - first
chat diagnosis said deadlock, CPU evidence corrected it to thrash. One
commit briefly carried a red test via a masked exit code; caught and
amended. Primary login node died mid-phase; login2 reroute. A100-first
routing policy discovered, measured, committed.

**Open.** T (Phase 2) now carries the validity case. Sensitivity set
(pooling variants, fixed-depth layer rule, n=300 vs 1500) not yet run -
scheduled with the tournament. A1 capacity-gate addendum pending (needs
probe runs + track decision per N5). OLMo/Falcon3 M5-universality wants an
explanation before the paper.

**Gate P1->P2: PASSED.** Phase 2 (behavioral T) is open.

## 2026-08-04 — Phase 3 assembly notes, written BEFORE arm components land

**Correction of the in-chat "first look" (per-pair q_T means).** The
exploratory per-pair summary showed WA-C below WA-G and B on q_T, 6/6.
The FROZEN aggregation (prereg §3.1: B's pool over pairs per language,
then one T_ℓ; ε_B provenance was built on language-level cells) REVERSES
it: WA-C q_T median 0.19–0.29 across seeds; WA-G 0.0; B 0.0–0.006. No
prereg number was scored off the per-pair form; the frozen form governs
all scoring from here.

**Mechanism (measured, not theorized):** language-level text-benefit
denominators are nearly equal across the trio (median 0.35–0.38 nats).
The flip is in the numerator: WA-C's language-level injection benefit is
+0.084 nats; WA-G's is NEGATIVE (−0.045); B ≈ −0.006; base ≈ +0.006. The
collapsed arm benefits MORE from injected factor states than sound arms
do. Candidate reading — impoverished internal representations lean harder
on any informative context state — is recorded as a hypothesis, not a
claim. Pair-level and language-level T disagree on the arm ranking; this
aggregation sensitivity is a paper finding regardless of S1's outcome.

**Declared before assembly of any arm RMFS:** language pool for arm-level
comparisons = median over defined languages (primary), mean beside it
(sensitivity), undefined excluded and counted. Committed in
`src/rmfs/metrics/rmfs.py` and the assembly manifest before job 1706836
drained.

**S1 consequence:** the q_T leg alone no longer sinks WA-C; S1 now rests
on q_C (C_pres) pulling WA-C's soft-min below WA-G's — while WA-G's own
q_T ≈ 0 floors ITS score. Both scores sit near the soft-min floor;
the verdict is genuinely undetermined until arm components land. Both
outcomes remain reportable under the frozen protocol (S7 branches).

**First RMFS numbers in existence** (observational, frozen form, τ=0.1):
Qwen3-0.6B 0.108 / Qwen3-4B 0.165 / OLMo-2-1B 0.063 / bloom-1b7 0.069
(median over languages; component vectors in results/tables/rmfs_v1.*).

## 2026-08-04 — S1 scored: FAIL on primary pool (frozen protocol, no discretion)

Job 1706836 (76/76 complete, zero failures) supplied arm q_L and C_pres;
assembly ran the frozen form (τ=0.1, language-level T, median pool
declared pre-assembly).

**Verdict: S1 FAIL. WA-C ranks ABOVE WA-G and B on median-pooled RMFS in
3/3 seeds** (WA-C 0.152–0.167; WA-G 0.1258 all seeds; B 0.126–0.129).
Sensitivity: under the MEAN pool S1 would pass (C<G 2/3, C<B 3/3) — the
verdict is pool-dependent because both arms sit at the soft-min floor.
The declared primary governs: FAIL.

**Mechanism, exact:** at τ=0.1 the soft-min is ≈ min(q). WA-G's minimum
is q_T = 0.000 (language-level injection benefit ≤ 0 in ≥ half its
languages); WA-C's minimum is q_C ≈ 0.049–0.058. The collapse detector
WORKED — C_pres separates sound (1.00) from collapsed (0.05) by ~20x,
in line with the signature expectation. The behavioral component is the
culprit: language-level T_inj reads sound arms as zero and collapsed
arms as positive.

**Systematic pattern across all WA arms:** collapsed-family ckpts
(WA-C/D/F, q_C ≈ 0.05) all show POSITIVE q_T medians (0.19–0.33, zero
saturated languages); sound ckpts (WA-B/G, q_C = 1.0) all show q_T = 0
with 6–16 saturated languages; WA-E intermediate on both (q_C ≈ 0.27,
q_T = 0). 9/9 vs 6/6 consistency. The pre-registered hypothesis stands
strengthened: T_inj as operationalized rewards context-dependence, which
anti-correlates with representational health across these arms. This is
a finding about behavioral transfer metrics, not a bug in the harness
(identity-patch validation exact; A4 smoke passed its frozen bar).

**Protocol consequences:** S1 outcome recorded as scored — no redesign,
no re-aggregation (rule: failures are findings). The S7 instrument
branch (components as fault detectors: C_pres 20x separation, q_T's
inverted signal itself diagnostic) remains open and is now the live
claim path. Battery re-score (S2, synthetic injections vs 75-cell
signatures) and the validity tournament proceed as frozen.

## 2026-08-04 — S2 battery scored: 49/75 cells; the panel's character is now measured

Jobs 1707443 (geo; re-emitted as 1707563 for the invariant track) and
1707445 (T; 40,800 rows). Two scorer defects found and fixed between
first and final scoring, both ledgered in-line: q_K numeric cells were
compared on rung INDEX not value (flipped pairing-permutation q_K to its
correct PASS), and the rotation-global q_L cell was scored on the
z-track when its own frozen note pins it to the invariant track (re-run
confirms: PASS on the correct track). Final: **49/75, 0 unscored.**

Per component: C_pres 13/15 · q_K 12/15 · q_L 11/15 · RMFS 10/15 ·
**T 3/15**.

**What the battery proves works:**
- C_pres reads uniform collapse EXACTLY — (1−m)² within 0.01 at all
  three m levels and for the single-language collapse. The founding
  blindness result reproduces as frozen: LFS-VC band-flat under all
  collapse levels while C_pres falls 99%.
- The ladder escalates where escalation is mandatory (per-language
  rotations θ=0.4/0.8 → ≥M3 ✓; permutation → M5 with q_K crashing ✓)
  and the invariant track is genuinely rotation-invariant.

**S2 verdict (strict, per signature matrix rows): FAIL — 12/20 collapse
cells.** Every failed collapse cell is a T or T-inherited-RMFS cell (plus
one frozen cell that contradicts its own note, below). Detection by
C_pres: 4/4 exact. Both statements are the record.

**The T instrument characterization (the battery's central finding):**
- T requires a valid factor: destroying concept correspondence outright
  (pairing permutation) drives T to +0.004 ≈ 0 — its strongest response,
  missing the 3×floor bar by 0.0002 (k=3 is frozen; FAIL stands).
- T is HEALTH-INVERTED among damaged-but-alive worlds: mild collapse
  RAISES T above base (+0.019 → +0.035 at m=0.1), and T declines
  monotonically with severity (+0.030 at 0.5, +0.013 at 0.9). Same
  signature that ranked WA-C above WA-G in S1, now dose-controlled.
- Reading: T ≈ (context dependence) × (factor validity). It nulls when
  the factor is garbage but inflates when representations are
  impoverished. As a health gauge it is disqualified; as a diagnostic
  its inversion is itself informative. (Estimator note N7.)

**Frozen-table defects the battery exposed (E-ledger E4):**
- offset q_K cell demands rung ≤ M1 post-injection, but the BASE world
  selects median M2 — unsatisfiable as written; conflates "fault absorbed
  at low rung" with "absolute rung is low."
- collapse_per_language q_L cell says "up" while its own note derives
  q_L DOWN ("language share rises"). Scored as frozen: FAIL recorded.
- pairing-permutation C_pres band assumed a per-language variance
  estimator; the declared pooled-REML estimator correctly reads
  correspondence destruction as concept-variance death (0.001). Estimator
  sensitivity recorded, declared choice governs.
- C_pres sentence-sampling floor is wide (SD 0.25 across resampled
  worlds): band cells on C_pres are lenient. Exact cells are unaffected
  (fixed 0.01 tolerance) and carried the S2 detection verdict.

**Genuine geometric findings:** low-rank C×L interaction leaks into q_L
(0.27 → 0.13); my declared warp magnitude (2.0) destroys real variance
(C_pres 0.10), overshooting the "band" expectation; the one-SE ladder
under-escalates for shear and interaction at n=300 (M2/M3 vs mandated
M4) — selection power, not absorption. S3 (stability): RMFS invariant-
track cell PASSED. S4 strict: FAIL (same T-dominated pattern; magnitude-
recovery calibration curves not yet run — open item).

## 2026-08-04 — Tournament core scored (S5/S6/S7): the panel's validity map

Local run, frozen protocol (raw-residual deflation, language-cluster
bootstrap 2000, seed 13, Romano-Wolf FWER over the 11-candidate family).
Tables: results/tables/tournament_core.csv, tournament_rw.csv.

**S5 (predict held-out content transfer, 3.1.3 set): PASS — both lines.**
rmfs_obs ρ=0.634 [0.510,0.757] p<0.001 (bar ≥0.30); the pre-registered
circularity-free T-excluded aggregate ρ=0.679 [0.594,0.826]. The
geometric gauges are STRONG content-transfer predictors: q_L ρ=0.763 —
numerically the best single predictor in the study, statistically tied
with shared-prep MEXA (0.762); q_K ρ=0.548. And q_T ρ=−0.418: NEGATIVE,
converging with S1 and the battery dose-response — three independent
designs now agree T is health-inverted.

**S6 (per-language signal beyond observables vs alpha): PASS.**
rmfs_obs deflated ρ=0.244 [0.075,0.464], p=0.008, Romano-Wolf p=0.019 —
survives family-wise correction, statistically tied with shared-prep
MEXA (0.243). Caveat recorded: rmfs_obs lives on the 4-model/147-row
support; texcl (14 models) is null vs alpha (0.042), as are q_L/q_K/LDE
individually — a same-support texcl comparison is pending before any
"aggregation adds alpha-validity" claim is made.

**S7 (increment beyond shared-prep MEXA): first branch FAIL** —
orthogonalized rmfs_obs ρ=0.074 [−0.108,+0.252]. Per the frozen either/or,
the claim is the second branch: unique fault detection where MEXA is
flat (S1 components + S2 battery, already on record).

**Structural finding — Grinold effective breadth ≈ 1.2 (texcl) and 2.0
(MEXA) languages.** The per-language panels of EVERY candidate collapse
to ~1-2 independent bets (q_K is nearly constant across languages; MEXA
per-language values are heavily cross-correlated). Per-language validity
claims in this literature are structurally weaker than their nominal n
suggests — reframes 3.1.4 comparisons generally, ours included.

Incumbents for the record: legacy-grid MEXA 0.306 (RW p=0.003) remains
the strongest alpha predictor; AaR 0.209. CKA and probe-MDL: NOT
COMPUTED (no grid pass exists) — blank, not invented. MCS and PBO/CSCV:
pending (need the config-variant grid). Scorecard so far:
S1 FAIL · S2 strict FAIL (C_pres detection 4/4) · S3 PASS · S4 strict
FAIL · S5 PASS (both lines) · S6 PASS · S7 instrument branch · S8-S10
pending (walk-forward blocked on HF token rotation - user action).

## 2026-08-05 — A5 arms downstream exam scored: P1 FAIL, the premise falsifies

75/75 Belebele + 75/75 baselines, zero task failures. Predictions scored
verbatim (A5 frozen 2026-08-04, before any score existed):

**P1 FAIL 0/3.** WA-C (89% concept collapse at L8) scores 0.389-0.402
panel_acc — indistinguishable from control B (0.383-0.398), slightly
ABOVE it in all three seed-matched comparisons. **The C_pres false-alarm
clause binds: C_pres flagged a model whose downstream ability was NOT
degraded.** P2/P3 void. **P4 PASS** (WA-G − B = +0.0075 ≤ 0.02):
alignment training bought no downstream multilingual gain.

**Arm-panel validity (Spearman vs panel_acc, family-cluster boot):**
AaR +0.505 [+0.093,+0.713] — the WINNER; MEXA +0.361 [−0.035,+0.648];
panel_stat −0.532 [−0.805,−0.143] — significantly NEGATIVE; rmfs_int
−0.378; q_L/C_pres/q_T individually null. Inherited q_K constant (no
rank signal by construction).

**Mechanism (composition table, a5_arm_panel.csv):** the damage axis in
this population is NOT collapse. Worst performers (0.326-0.336) are the
representational-objective arms D/E/F/G_tail — whose geometry looks
CLEAN (C_pres 0.87-1.00). Best performers (0.394-0.399) include the
three collapsed WA arms (C_pres ≈ 0.05). Downstream harm came from
objective-gradient interference, which AaR's tail statistic partially
tracks and our variance-structure gauges do not; concept collapse at
the dip layer was behaviorally benign. MEXA scores the collapsed arms
HIGHEST (0.586 vs 0.48 controls) — its collapse-affinity is confirmed
again, but on this task the affinity carries no downstream penalty.

**What this falsifies:** the founding premise "collapse = broken model,"
inherited from the LFS demotion, is FALSE at the behavioral level for
0.6B/Belebele-MCQ. The fault-detection value proposition must be
restated: the panel detects representational REORGANIZATION — real,
large, and invisible to nothing else — but reorganization does not
imply harm; harm prediction on this population belongs to AaR.

**Scope limits (recorded, not excuses):** one task family (in-language
MCQ discrimination), one scale (0.6B), gauge at L8. Arm t_rows show
WA-C's language-level text benefit (0.35 nats vs 0.38 healthy) barely
reduced — the generation-side content channel was also largely intact.

**Standing after A5:** the beat-MEXA quantitative claim is DEAD on this
exam; AaR is the strongest metric on fault-containing panels and MEXA
is adequate. The panel's live value: (i) q_L's content-transfer
validity 0.76 (S5, healthy grid) stands; (ii) the reorganization-
without-harm finding is novel and publishable (how does a model with 5%
concept variance at L8 keep full task ability? — connects to N5); (iii)
the validation harness itself, which has now killed three of its own
headline claims honestly. Every failure is on the record beside every
pass.

## 2026-08-05 — 14-model same-support rerun: the scalar's alpha signal was
## small-support artifact; the three-channel architecture is confirmed

Sequence note: A6 (channel specification) was committed BEFORE this
rerun's results existed.

With T extended to all 14 grid models, the frozen v1 scalar's deflated
alpha correlation collapses: 0.244 [0.075,0.464] at 4 models → **0.039
[−0.045,+0.107], RW p=0.648, at 14**. The S6 PASS stands as scored on
its frozen support and does NOT generalize across the grid; recorded as
exactly that. q_T at 14 models: −0.011 vs alpha (null), −0.266 vs
content (still negative).

Unchanged pillars: mexa_shared 0.243 / mexa_leg 0.306 on alpha (only
RW-significant intrinsics); q_L 0.763 / texcl 0.679 on content
transfer; AaR +0.505 on the fault panel (A5). Face validity of the
14-model table: multilingual-focused models (Mistral 0.224, EuroLLM
0.207, salamandra 0.196-0.203) top; English-centric (Falcon3 0.033,
OLMo-2-7B 0.040) bottom.

**Consequence:** no single quantity carries more than one exam — the
channels are complementary and non-redundant, which is the entire A6
design. RMFS (the instrument: structure + discrimination + tails +
reorganization flag) is the deliverable; every channel keeps its own
validated lane. Remaining: blind walk-forward W1-W4 (blocked on HF
token rotation, user action).

## 2026-08-05 — framing correction + clustering robustness on F5

**Framing (corrected).** External review flagged that "LFS was failing"
overstates and contradicts our own memo ("LFS was not discarded"). The
precise statement, verified against the frozen record: the collapse
boundary of the SCALAR reporting form was pre-registered
(PREREG_BATTERY 2026-07-15, I8 row + CVP as named detector + "effective
rank alone is an insufficient anti-Goodhart guard"; PREREG_312_313
2026-07-07, P-A2.3 "the critical one", committing to report the naive
form falsified if triggered), reproduced in trained models, and
triggered a pre-committed branch to channel reporting. LFS's
decomposition is not the casualty — it is the strongest channel
(ρ=0.76). All three documents rewritten accordingly; no numbers changed.

**Clustering robustness on the 0.76 (new computation).** Motivated by
F9 (effective breadth ≈ 2), the content-transfer estimate was re-run
under coarser clusterings: language 38 clusters [0.674, 0.884];
macro-family 10 clusters [0.604, 0.889]; model 4 clusters
[0.124, 0.837], p=0.042. Conclusion recorded: the estimate is robust to
cross-language correlation; the binding uncertainty is the MODEL
dimension (n=4 in the 3.1.3 set), and the k=4 bootstrap is itself in
few-cluster territory. Published with the estimate from now on; the
walk-forward is the designed remedy.

## 2026-08-05 — continuity check against the July email to Koehn

Verified the two claims made in `~/Desktop/multilingual-metrics/paper/
KOEHN_EMAIL_DRAFT.md` (July): (a) "predicts downstream performance at
about 0.59 after deflation on exam-capable models (ten models, six
independent families)"; (b) "two never-touched model families passed a
frozen walk-forward blind, four predictions of four" (granite-3.1-8b,
Yi-1.5-9B; predictions P-A3.1-P-A3.4 in the legacy battery prereg).

**Re-derivation on the current stack, same exam-capable rule (mean
0-shot Belebele >= 0.30, 10 of 17 models qualify):** MEXA at the
LFS-selected layer vs alpha = **0.487 [0.365, 0.554]**, against the 0.59
reported in July. Full-panel value 0.306. The stored artifact behind the
July 0.59 is not present in the legacy deflation outputs, so the cause
of the gap (estimator scheme within the published bracket, alpha
shrinkage, or language set) is NOT yet determined — recorded as open,
NOT as a resolved revision. Disclosed to Koehn in the current email
rather than left silent.

Walk-forward continuity: the July run validated the LFS-era
configuration on granite/Yi; the queued run tests the channel form on
Gemma-3-4B-pt and Aya-Expanse-8B (untouched, predictions frozen in A6).
Both statements now appear in the email so the two messages cannot read
as a retraction.

## 2026-08-07 — layerwise profiles landed (12/14 models)

Job 1709884: 38 tasks COMPLETED, 2 TIMEOUT, 2 CANCELLED -> 12 of 14
models profiled at every dumped layer. Missing: Mistral-7B-v0.3,
OLMo-2-1124-7B (resubmit pending).

**Result (serves the research objective's "across Transformer layers"
clause, previously unaddressed):** every model shows the same U shape in
the language share — high at the embedding (0.93-0.98), minimum in the
middle-to-late layers, rising again at the output (0.94-0.97). What
separates models is DIP DEPTH, not shape: Qwen3 reaches 0.59-0.71 at its
minimum, salamandra 0.65-0.78, while bloom/SmolLM2/OLMo-2-1B barely dip
(0.88-0.90). Within the Qwen3 family the dip deepens with scale (0.6B
0.71, 1.7B 0.71, 4B 0.59, 8B 0.60). The language component is therefore
a function of depth, not a per-model constant — and the single-layer
numbers reported elsewhere are the minimum of that curve.

## 2026-08-07 — A7 translationese test scored: T1 FAIL, T2 FAIL, T3 PASS, T4 PASS

Job 1749800, 14/14 models, zero failures. WMT19 both directions, same
language and domain, 300 pairs per language per direction, 7 languages.

**T1 FAIL (2/7, bar 5).** Sentence matching is NOT systematically
higher on translationese. It is LOWER in 5 of 7 languages
(zh -0.047, gu -0.046, lt -0.043, kk -0.038, de -0.004) and higher only
in ru (+0.067) and fi (+0.004). My stated reason for expecting the
opposite — translations track their source more literally — is wrong
as a general rule at the sentence-embedding level.

**T2 FAIL (0/7, bar 5).** The language share is HIGHER on translationese
in all seven languages (+0.020 to +0.056), the opposite of the frozen
prediction. Reading: translated text carries more of the source
language's structure, which the decomposition scores as MORE language
signal, not less. Direction was predicted backwards; the effect is
consistent and worth reporting on its own.

**T3 PASS.** Model ordering is preserved almost exactly: Spearman
between per-model readings on native and on translated text is
matching 0.991, language share 0.996, tail 0.947 (bar 0.80).

**T4 PASS.** The native-to-translated gap is an order of magnitude
smaller than the spread across models: matching 0.016 vs 0.231;
language share 0.032 vs 0.154; tail 0.047 vs 0.186.

**Consequence.** Translationese shifts the LEVEL of the readings by a
small, systematic amount and leaves the COMPARISON between models
intact. The project's conclusions — which are all comparative — survive
the corpus dependence. This is the first direct evidence on the
question in this project, and it closes the one Aim 1 commitment
(evaluation that avoids translationese) that had never been tested.
Both failed predictions are reported as scored; the T2 direction
reversal is a finding in its own right.

## 2026-08-07 — A8 native exam scored: N1 PASS, N3 PASS, N2 FAIL, N4 FAIL
## — and the control shows the cause is NOT translation

Job 1749915, 14/14 models, 30 natively authored exam languages
(INCLUDE), zero failures.

**N1 PASS** (12/14 models above chance; EuroLLM-1.7B 0.246 and
bloom-7b1 0.244 at chance, excluded by the standing rule).
**N3 PASS** (model ordering on native vs translated exams,
Spearman 0.982).
**N2 FAIL**: matching predicts native exam performance at +0.080
[-0.024, +0.152] — interval includes zero.
**N4 FAIL**: 0.080 is outside the translated exam interval
[0.172, 0.289].

**Control (run before interpreting):** the TRANSLATED exam restricted
to the identical 30 languages and 14 models gives matching +0.073
[-0.028, +0.154], mexa_leg +0.118, q_L +0.018 — statistically
indistinguishable from the native exam values. **The drop from 0.243 to
0.080 is therefore attributable to the language panel, not to
translation.** Translationese did not inflate the reported validity.

**Mechanism (supported, not proven):** the exam subset covers only
languages with formal examination systems, which are well resourced.
Resource spread collapses from SD 1.67 (full 69 language panel, range
3.8 to 12.5 log tokens) to SD 0.92 (30 language subset, range 7.3 to
11.9); the entire low resource tail (Somali, Guarani, Haitian Creole,
Yoruba, Sundanese, Swahili) is absent. Range restriction of this size
is sufficient to explain the loss of signal.

**Consequence for the 3.1.4 claim, restated:** intrinsic alignment
measures predict downstream performance ACROSS a wide resource range;
WITHIN a band of similar, well resourced languages no measure tested
here predicts it, ours or the incumbent's (matching 0.073-0.080,
mexa_leg 0.118-0.139, q_L 0.009-0.018, all intervals touching zero).
This is a scope condition on the whole class of metrics, not a defect
specific to RMFS, and it sharpens the earlier effective breadth
finding. The content transfer result (0.76) is measured on the full
panel and is unaffected.

## 2026-08-08 — why the narrow panel fails: the target is only ~31%
## reproducible, and averaging exams partially recovers the signal

Having two independent exams over the identical 420 cells (14 models x
30 languages) makes the target's own reliability measurable for the
first time in this project.

- Raw accuracy agreement between the translated and native exams:
  **rho = 0.886** — both measure ability well.
- **Deflated residual agreement: rho = 0.315.** After removing model
  effects, training data volume, family, script and fertility, two
  independent exams of the same construct agree on only about a third
  of the remaining per-language rank variation. The rest is exam
  specific.

**Implication.** Treating the two exams as parallel forms, the ceiling
on ANY predictor of one exam's deflated residual is sqrt(0.315) =
**0.56**, not 1.0. The reported 0.073-0.080 is low against that ceiling
but is not being compared against a fair one when quoted against 1.0.
Disattenuated, the matching reading gives 0.143 on this panel.

**Confirmation by construction.** Averaging the two exams raises target
reliability to 0.479 (Spearman-Brown) and the measured correlation
rises accordingly, from +0.073 / +0.080 on either exam alone to
**+0.101 [+0.003, +0.175]** on the average — the interval now excludes
zero on the same 30 languages where each exam alone was null. The
signal was partly hidden by target noise, not absent.

**What this changes.** The A8 verdicts stand as scored (N2, N4 FAIL).
The interpretation is now supported by measurement: on a narrow
resource band, per-language downstream ability is weakly reproducible
across instruments, which bounds what any intrinsic metric can achieve
there. The practical remedy is a more reliable target (pooled
benchmarks), not a different representation method.

## 2026-08-09 — A9 scored: G1 PASS, G3 PASS, G2 FAIL (wrong dimension
## expanded), and the underlying question answered separately

Jobs 1750325 (readings) and 1750357 (both exams). 19 of 21 new models
completed; internlm2_5-7b and EuroLLM-9B excluded for technical and
licence reasons respectively. Grid is 33 models, not 35.

**G1 PASS, 19/19.** The U shaped depth profile reproduces in every new
model: language share highest at the embedding and the output, minimum
in between, typically at 40-55 percent of depth. Across all models
measured to date this is 31 of 31. Two anomalies worth naming:
TowerBase-7B and occiglot-7b-eu5 both take their minimum at layer 2
(6 percent of depth), unlike anything else in the grid; both are
European focused continued pretraining recipes.

**G3 PASS.** On the expanded panel (2,001 rows, 33 models, 73
languages) with the pooled two exam target, the matching reading gives
**+0.238 [+0.181, +0.278]**, the tail reading +0.105 [+0.051, +0.148],
and the meaning-against-language reading +0.041 [-0.040, +0.122].
Validity survives near tripling the model count, and its interval is
tighter than the 14 model estimate.

**G2 FAIL, and the reason is my error, not the metric's.** G2 asked for
the model clustered interval on the content transfer result to narrow
by 25 percent. It did not move at all (0.712 vs 0.713) because the
legacy content transfer target exists for only 4 models; expanding the
grid added exam data and no content transfer data. The dimension that
binds that particular estimate was untouched by the expansion. Scored
FAIL as frozen.

**The underlying question, answered by a separate analysis (labelled
exploratory, not a substitute for G2).** Our own generation harness
measured content transfer for all 14 original grid models, which the
legacy file did not. Recomputing the same relationship on that data:
meaning-against-language **+0.644 [+0.383, +0.765]**, matching
**+0.706 [+0.470, +0.843]** — interval widths 0.383 and 0.373 against
0.713 for the 4 model legacy estimate, a **46 percent narrowing**. The
0.76 was not fragile; it was imprecise for want of models, and with 14
it is both smaller and far better determined. Extending the harness to
the 19 new models would settle it at 33.
