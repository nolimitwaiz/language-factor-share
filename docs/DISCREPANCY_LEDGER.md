# Discrepancy Ledger

Every divergence between a reported value and a reproduced one, every data
integrity problem, and every unexplained result. Entries are appended, never
edited away.

Format: what was expected, what was found, the exact code path, a
hypothesis, and a classification: (a) environment or version drift,
(b) unlogged config difference, (c) genuine bug in the original,
(d) unknown.

---

## D1. Torn line in the run ledger (2026-07-24)

**Expected.** `ledger.jsonl` contains one complete JSON object per line.

**Found.** The cluster ledger contains 194 lines, of which 193 parse. Line
8 is the fragment `start"}`, the tail of a record whose head was
overwritten.

**Code path.** `src/common/ledger.py:_append`, as originally written:

```python
with open(LEDGER, 'a') as f:
    f.write(json.dumps(rec, default=str) + '\n')
```

**Hypothesis.** Python's buffered writer may split one logical write across
multiple `write()` syscalls. Concurrent SLURM array tasks (up to 20 running
simultaneously during the budget campaign) append to the same file with no
lock, so two writes can interleave and tear a record.

**Classification.** (c) genuine bug in the original, in infrastructure
rather than in a scientific code path.

**Impact on results.** None. The lost record is a run-start marker; its
matching run-end record is intact, and no analysis reads the ledger.
Provenance for one run start is unrecoverable.

**Fix.** `_append` now takes an exclusive advisory lock (`fcntl.flock`),
flushes, and fsyncs. Verified by writing and parsing. Records written before
this fix are otherwise intact and were not modified.

---

## D2. Stale prototype result in a campaign directory (2026-07-24)

**Expected.** Every `results/budget/<model>/budget.json` holds an n=1500
campaign result.

**Found.** `results/budget/OLMo-2-0425-1B/budget.json` held an n=300
prototype result, distinguishable only by `selected_rank: 16`, a rank that
does not occur at campaign scale. The campaign run for this model timed out
twice (36 hours, then 60 hours) and was never rerun.

**Code path.** `src/analysis/budget_real.py` writes to a path keyed only by
model tag and configuration suffix, so a later run at a different `n`
overwrites, or fails to overwrite, silently.

**Classification.** (b) unlogged config difference, promoted to a data
integrity risk by the path scheme.

**Impact on results.** None on the report, which excludes this model from
the campaign primaries and says so. A reader recomputing summaries by
globbing `budget.json` would silently mix scales; one analysis script
already guards on rank, which is how this was caught.

**Fix.** Renamed to `budget.STALE_n300.json` on both the local and cluster
copies, with a `README.md` in the directory. Globs now miss it and fail
loudly. Nothing was deleted. The path scheme itself should carry `n` in
future runs.

---

## D3. "Size-invariant by construction" was stated too strongly (2026-07-24)

**Expected.** The report asserted that variance shares are size-invariant by
construction (Figure 5 caption, and two further places), and that LFS "moves
only with the language set."

**Found.** Nearly true, but not exactly. Expanding the grid as
`x[c,l] = C[c] + G[l] + eps[c,l]` gives

```
SS_lang    = (L-1) D sigma^2 + N  * SS_G
SS_concept = (N-1) D sigma^2 + L  * SS_C
```

so the numerator carries a noise term that does not vanish, and under a
pure-noise null the estimator returns the degrees-of-freedom ratio
`(L-1) / ((L-1) + (N-1))` exactly, not zero. On real campaign
representations with a fixed 12-language set, LFS moves by -0.006 to -0.010
from N=100 to N=1000 (twenty repeats), in the direction the expansion
predicts.

**Code path.** `code/pilot_metrics.py:lfs`. The implementation is correct;
the claim about it was too strong.

**Classification.** (c) an error in the original, in the stated claim rather
than in the code.

**Impact on results.** No reported comparative claim changes. The noise term
is common to any two conditions measured at the same L and N, so it cancels
in dip depth, in cross-model comparison on a fixed language set, and in
before-and-after injection contrasts, which is the form of every comparative
claim in the report. What changes is the interpretation of absolute values:
the pure-noise value is 0.298 at the development grid (128 languages, 300
sentences) and 0.078 at the campaign grid (128, 1500), so absolute LFS values
from the two grids must not be placed side by side. MEXA's measured size
dependence is ten to nineteen times larger than LFS's on the same draws, so
the comparative point the claim was making survives.

**Note.** The pure-noise value is not a lower bound. A grid with strong
concept structure and weak language structure measures below it. An earlier
version of the unit test asserted otherwise and failed; the test's assumption
was wrong and was corrected, the metric was not touched.

**Fix.** Section 5.3 of the report rewritten to state the exact null value,
its two verifications (synthetic across nine configurations, and a
monolingual null on saved representations reproducing it to three decimals),
the measured real-data size dependence, and the three consequences. Three
further assertions elsewhere in the report scoped to match. Unit tests added
in `tests/test_estimator_floor.py`. Consequence recorded in the Aim 2 plan:
an objective that minimizes LFS pushes against an asymptote set by the
residual scale, not toward zero.

---

## D4. Permutation-null gate written with an inverted inequality (2026-07-27)

**Expected.** `prereg/PREREG_WORDLEVEL.md` G-WL.3 required that real
word-level LFS *exceed* the 95th percentile of a permutation null in which
alignments are shuffled within sentence pairs.

**Found.** The real value is far *below* the null: 0.070 against a null mean
of 0.312 (primary) and 0.057 against 0.151 (secondary).

**Why the gate was wrong.** Shuffling the concept assignment destroys
concept structure. That shrinks the denominator of
`V_lang / (V_lang + V_concept)` and drives LFS *up*. A signal-present result
is therefore a value far below the null, not above it. The implementation is
correct and the data pass the scientific check by a factor of four to five;
only the inequality written into the pre-registration was backwards.

**Classification.** (c) an error in the original, in a pre-registered
threshold rather than in code.

**Impact on results.** None on any number. The gate is reported as
mis-specified rather than as failed, and the correct-direction result is
reported alongside.

**Pattern worth noting.** This is the third instance of the same underlying
mistake: predicting the direction of LFS while reasoning only about its
numerator. The earlier two were the collapse cell (I8, predicted LFS would
fall, it rose) and the permutation cell at sentence level. Any future
prediction about LFS's direction must reason explicitly about both terms of
the ratio, and a note to that effect now heads the prediction section of the
word-level pre-registration.

---

## D5. Function words contaminated the word-level concept grid (2026-07-27)

**Expected.** `prereg/PREREG_WORDLEVEL.md` specifies the concept unit as an
English word occurrence "restricted to content words as defined on the
English side."

**Found.** The Stage 1 concept grid contained 35 to 39 percent function
words. The most frequent "concepts" in the four-language grid were `and`
(93 occurrences), `in` (63), `with` (22), `on` (19), `to` (18), `but` (17),
every one of which is in the project's own stopword list.

**Code path.** `src/wordlevel/align.py:one_to_one` filters links for
one-to-one correspondence but never applied `is_content`. The content filter
existed only inside `coverage()`, which is a reporting function, so coverage
was reported correctly for content words while the link set used downstream
contained everything.

**Classification.** (c) an error in the original: an implementation that did
not match its own pre-registration.

**Why it mattered.** Function words were separated out deliberately, and
Stage 0 had already measured that they align worse than content words in
every one of twenty languages. They are also more grammaticalized and
therefore expected to carry more language-specific information. Including
them inflated the measured language share.

**Impact on results.** The first Stage 1 run is superseded. Re-running with
the filter applied changed the picture materially at one of the two
operating points:

| | concepts | pure-noise value | word LFS at dip | headroom |
|---|---|---|---|---|
| primary, unfiltered | 252 | 0.0420 | 0.0697 | 1.7x |
| **primary, content only** | **154** | **0.0671** | **0.0633** | **0.94x** |
| secondary, unfiltered | 1316 | 0.0030 | 0.0569 | 19x |
| **secondary, content only** | **851** | **0.0047** | **0.0533** | **11.4x** |

Removing function words cut the primary point's concept count by 39 percent,
which raised its pure-noise value from 0.042 to 0.067, while the measured
value fell to 0.063. **The primary operating point now sits at or below the
estimator's pure-noise value and is uninformative**, which is one of the
conditions the pre-registration named in advance. The secondary point
retains eleven-fold headroom and remains interpretable.

**Fix.** The filter is applied where the concept set is defined
(`stage1.shared_concepts`), not in the aligner, so alignment files continue
to hold all agreed links as raw output while the experiment decides which
links are concepts. Unfiltered results retained as `UNFILTERED_*.json`.

---

## D6. The Aim 2 CVP criterion is failed by the control arm

**Discovered.** 2026-07-27, on the first scoring pass over the Aim 2 study,
seed 0.

**What the pre-registration said.** `AIM2_STUDY_PLAN.md` section 7, criterion
2: an arm is promising only if `CVP >= 0.90` against the frozen model
*throughout training*.

**What happened.** Arm B, ordinary continued language-model training carrying
no alignment term, no CVP term and no LFS term, breached the floor at 2 of 21
logged steps, reaching 0.855. Arm B is the control. A criterion that the
control fails cannot separate a damaging objective from an undamaging one,
which is the entire job it was written to do.

**Classification.** (b) a criterion that was mis-specified when frozen. The
threshold and the instrument are both sound; the quantifier is not.
"Throughout training" was written as if concept variance were monotone under
training, and it is not. It fluctuates step to step under plain
language-model training alone, so a per-step universal quantifier converts
ordinary sampling noise into a failure.

**This is the second time a frozen gate has been mis-specified in the same
way** (see D4, the inverted permutation gate). Both errors came from
reasoning about the intended direction of an effect without first asking what
the quantity does under the null. The habit to correct is to run the control
before freezing the rule, not after.

**What is NOT being done.** The criterion is not being loosened to make arms
pass. Seed 0 is scored against the frozen rule exactly as written, and every
arm including B is recorded as failing criterion 2. Changing the rule after
seeing which arms it rejects is the precise failure the pre-registration
exists to prevent.

**Proposed replacement, to be frozen before it is applied and applied only
prospectively.** Three changes, each with a stated reason:

1. Evaluate CVP on the *final* checkpoint rather than at every logged step,
   since the deliverable is a trained model and transient dips during
   training are not a property of it.
2. Add a separate trajectory criterion scored *relative to arm B* rather
   than absolutely: an arm's mean CVP over training must be at or above arm
   B's mean minus its across-seed standard deviation. This keeps the
   discriminating power the absolute floor was meant to have, while not
   penalizing an arm for fluctuation that plain training also produces.
3. Retain the absolute floor as a *tripwire* for live monitoring, which is
   the role it actually performed well: it caught arm C's collapse to 0.449
   and arm D's to 0.287 while they were happening.

**Impact on results.** Under the frozen rule, no arm passes and the study
reports no working objective. Under the proposed replacement, the seed 0
ordering would be unchanged in the comparisons that matter: D remains the
most damaging arm at mean CVP 0.575, E remains 0.333 above it, and C remains
a collapse. The replacement is therefore not expected to change any
conclusion, which is the only condition under which changing a frozen rule
is defensible at all, and it still does not take effect for seed 0.

---

## D7. Criterion 5 was scored on sign alone and is uninformative as written

**Discovered.** 2026-07-28, on the first full scoring pass with downstream
results present.

**What the pre-registration said.** `AIM2_STUDY_PLAN.md` section 7, criterion
5: "Cross-lingual context use improves relative to Arm B." Criterion 4, by
contrast, says "improves relative to Arm B, **or holds within noise**", and
therefore carries an explicit noise band. Criterion 5 carries none.

**What happened.** Implemented literally, criterion 5 tests whether the
context-gain difference against arm B is greater than zero. The observed
differences are between -0.006 and +0.006 in the arm means, against a typical
per-language standard error of 0.047, roughly eight times larger. Every arm's
verdict on criterion 5 is therefore decided by the sign of a quantity that is
statistically indistinguishable from zero, and flipped between seeds for
arms C, E and F for that reason.

**Classification.** (b) a criterion mis-specified when frozen, compounded by
(c) an implementation that applied it literally rather than flagging that it
could not discriminate. The measurement itself is sound: the frozen model
reproduces the Aim 1 context-gain result across all six languages, and the
per-pair standard errors are correctly computed and reported. It is the
decision rule laid on top that carries no power.

**What is NOT being done.** Criterion 5 verdicts are left in the scorecard
exactly as the frozen rule produces them, including the passes that are
noise. They are marked as uninformative in the scorecard text rather than
recomputed under a rule invented after seeing the data.

**Proposed replacement, prospective only.** Give criterion 5 the same noise
band criterion 4 already has, and state the power calculation before running:
at 60 sentence pairs the per-language standard error is about 0.047, so the
design can detect a context-gain difference of roughly 0.10 and no smaller.
Either the pair count rises to the point where the effect of interest is
detectable, or the criterion is dropped as unmeasurable at this scale and
said so plainly. Adding pairs is cheap here and is the preferred fix.

**Impact on results.** None on the overall verdict. Every arm already fails
on criteria 2 and 6 at every seed, so criterion 5 changes no arm's outcome.
It is recorded because a criterion that returns a verdict without having the
power to support one is a defect whether or not it happened to matter.

---

## D8. The code-switch pre-registration mis-described its own held-out set

**Discovered.** 2026-07-28, after the code-switch arms trained and before any
downstream result was read.

**What the pre-registration said.** `PREREG_CODESWITCH.md` section 2.2:
evaluation "reports separately on the switch languages and on eight languages
that appear in no arm's training data in any form."

**What is actually true.** That claim is false for five of the eight. The
evaluation set inherited from the LFS study is Arabic, Bengali, Chinese,
French, Indonesian, Russian, Yoruba and Zulu. The code-switch corpus covers
fourteen languages including Bengali, Chinese, French, Indonesian and
Russian, and the replay mixture draws **full sentences** in all fourteen. So
five of the eight received direct multilingual exposure in every arm.
**Genuinely unseen by every arm: Arabic, Yoruba and Zulu, three languages.**

**Classification.** (b) an inaccurate statement in a frozen document,
introduced by reusing an evaluation grouping from a previous study without
re-checking it against a training corpus that had different language
coverage. The measurements are unaffected; only the label attached to a group
of them was wrong.

**Why it matters.** The held-out group exists to answer whether an effect
generalizes beyond the languages trained on, and whether continued training
forgets everything else. A group that is five-eighths trained-on cannot
answer either question. Read uncorrected, replay-exposed languages would have
been presented as evidence of generalization to unseen languages, which is
close to the opposite of what they show.

**Correction, applied to the analysis rather than to the frozen text.** The
prereg wording stands as written and wrong, with this entry attached. The
scoring reports three groups instead of two:

1. **switch-exposed**: languages whose words were substituted into the
   training text (German, Hindi, Swahili among the probed set).
2. **replay-exposed**: in the corpus and seen as full sentences through the
   replay mixture, but not probed during training (Bengali, Chinese, French,
   Indonesian, Russian).
3. **genuinely unseen**: Arabic, Yoruba, Zulu. Only this group speaks to
   generalization or to forgetting.

**Consequence for power.** The genuinely unseen group is three languages, not
eight, so any statement about generalization from this study rests on three
points and is correspondingly weak. That is a real cost of the error and is
reported alongside the result rather than buried.

**Procedural fix.** Evaluation groupings are now derived from the training
corpus manifest at scoring time rather than copied between studies. This is
the third error in this project traceable to reusing a rule without
re-checking its premises against the current setup (see D4, D6).

---

## D9. The concept-variance guard was inert in the word-level trainer

**Discovered.** 2026-07-29, when arms with and without the guard produced
bit-identical losses at two of three seeds.

**What happened.** `src/aim2/train_wordalign.py` computed the guard as

```
cv  = concept_variance(grid)
ref = cv.detach()
loss += w * clamp(CVP_FLOOR - cv / ref, min=0) ** 2
```

`cv / ref` is identically 1.0, so `clamp(0.90 - 1.0, min=0)` is always zero
and contributes no gradient. Arm WA-F was arm WA-D with dead code attached.

**Classification.** (c) an implementation error. The reference must come from
the frozen model, which `train.py` does correctly via a precomputed
`ref_grid`; the word-level trainer compared each batch against itself.

**How it was caught.** Not by a test. By noticing that two arms which differ
by a loss term reported identical numbers, and checking why. A guard that
silently does nothing produces plausible output, which is the failure mode
that unit tests on individual functions do not catch.

**Impact and correction.** The claim "the guard did not prevent the collapse"
was unsupported at the time it was made, because the guard was not running.
Fixed to use a frozen reference and re-run at three seeds. **The corrected
result reaches the same conclusion by a different route**: with the guard
verified active, concept variance still fell from 0.051 to 0.002. The reason
is structural rather than numerical, and is now stated as such in the report:
the hinge is bounded above by $(0.9)^2 = 0.81$ while the word-level term
starts near 600, so a bounded penalty cannot restrain an unbounded one at any
weight. This also explains why the weight sweep saturated between 100 and
1000.

---

## D10. The frozen reference was skipped by name in the evaluation scripts

**Discovered.** 2026-07-28, mid-run, from a single `[skip]` line in a log.

**What happened.** Both `evaluate.py` and `context_use.py` identified the
untrained model by the literal arm name `'A'`. The code-switch study names it
`CS-A`, so it fell through to the checkpoint branch, found no checkpoint, and
was skipped silently across all three seeds.

**Classification.** (b) a value correct in one study carried into another
without re-checking that it still applied. This is the same cause as D4, D6
and D8.

**Impact.** Criteria 1, 4 and 5 were unaffected, because they contrast against
the control arm rather than the frozen model. Criterion 6 and prediction P5
lost their baseline entirely and could not be computed.

**Correction.** The frozen reference is now identified by set membership
(`FROZEN_ARMS`) rather than by a literal. Only the untrained model was
re-run and merged, with a distinct output filename so it could not overwrite
the fifteen valid trained arm-seed points; those were verified intact before
the backfill was launched.

---

## D11. Headline profile counts combined distinct estimator cohorts

**Discovered.** 2026-08-20 during the ICLR submission reverification.

**Expected.** Working summaries referred to a homogeneous `31/31` set of
U-shaped LFS profiles and sometimes treated later expansion results as though
they were additional direct-SS measurements.

**Found.** The traceable original cohort contains 17 direct sums-of-squares
LFS profiles. The A9 expansion contains 19 REML LFS-VC profiles. Three models
occur in both cohorts, yielding 33 unique models, not one homogeneous 31-model
direct-SS experiment. An independent artifact audit found an interior minimum
with both endpoints higher in 17/17 direct-SS profiles and 19/19 REML
profiles. On 14 shared models the estimators agree strongly (Spearman 1.000,
Pearson 0.999987, maximum absolute value difference 0.00498), but they remain
mathematically distinct estimators.

**Classification.** (b) unlogged aggregation/wording drift in later summaries;
the saved measurements and estimator implementations are intact.

**Impact on results.** The central structural conclusion strengthens under
clean accounting: endpoint recovery is present in every profile within each
cohort. What changes is the headline wording and provenance. The profiles
also were not tested against a frozen smoothness or monotonicity rule, so
“interior dip with endpoint recovery” is the supported statement; a formal
smooth U-shape claim is not.

**Correction.** Report the two cohorts separately and state their three-model
overlap. Direct SS remains the primary paper estimator and REML LFS-VC is a
robustness analysis. See `docs/LFS_SUBMISSION_REVERIFICATION_2026-08-20.md`
and job `1758039`.

---

## D12. The 124/127 hub result is model-specific, not universal

**Discovered.** 2026-08-20 during the independent hub artifact audit.

**Expected.** Some report and summary language stated that the latent factor
beats English for 124 of 127 languages without attaching the result to a
model.

**Found.** The count ranges from 113 to 124 across the 17 audited models.
Llama-3.1-8B and several Qwen models reproduce 124/127, while BLOOM-1.7B is
113/127 and BLOOM-7.1B is 114/127.

**Classification.** (b) over-generalized reporting; the underlying hub
artifacts are unchanged.

**Impact on results.** The qualitative conclusion that a multilingual latent
factor usually outperforms English survives. The exact `124/127` sentence
must name the model and cannot describe the whole grid.

**Correction.** Report the across-model range and give named-model examples.
Canonical table: `rmfs/results/tables/lfs_hub_audit.csv`, job `1758039`.

## D13. Regression statistics quoted from the development-grid run (2026-09-10)

**Discovered.** 2026-09-10 while checking the third external review of the
ICLR draft against the stored artifacts.

**Expected.** The main text's confounder-adjustment sentence cites the
17-model grid, so its coefficients should match
`results/deflation/attribution/regression_summary.txt`.

**Found.** The draft (and `paper/TECHNICAL_REPORT.tex` near line 993) quoted
a token coefficient of +0.267 (SE 0.051, p 1.8e-7) and R^2 0.86. The stored
17-model artifact (n = 1,241 rows, 17 model fixed effects) gives +0.286
(SE 0.059, z 4.8), fertility -0.103 (SE 0.022), R^2 0.824. The quoted
values match the earlier 10-model development-grid run (1,173 rows).

**Classification.** (b) unlogged config difference: two runs of the same
regression on different model sets, with the earlier run's numbers carried
into later text.

**Impact on results.** Sign, magnitude, and significance are unchanged; the
exposure covariate remains the dominant predictor. No conclusion moves.

**Correction.** The ICLR draft reports the 17-model artifact values in the
setup section and in the regression appendix, and names the development-grid
run explicitly. The technical report sentence still carries the older run's
numbers with the 10-model grid context and is listed for the same update.

## D14. Offset-plus-scale range quoted from the three-model prototype (2026-09-10)

**Discovered.** 2026-09-10 while building the per-model decomposition table
for the ICLR appendix from `results/budget/*_fixed_raw/budget.json`.

**Expected.** "Offset plus scale account for 65 to 98 percent ... across the
14-model study" should be reproducible from the 14 common-rank artifacts.

**Found.** The 65 to 98 range is the three-model prototype at selected ranks
(Qwen3-1.7B r=16: 65.5; OLMo-2-1B r=16: 97.4; BLOOM-1.7B r=64: 76.2). Across
the 14 models at the common rank 64 the mean held-out offset-plus-scale share
runs from 62.9 percent (Qwen3-0.6B) to 97.2 percent (OLMo-2-1B); rotation
never exceeds 0.7 percent (BLOOM-7.1B 0.68).

**Classification.** (b) over-generalized reporting; the artifacts are
unchanged.

**Impact on results.** The conclusion that the cross-language map is
dominated by an additive offset and a scale is unchanged; the quoted range
moves by two points at the low end and one at the high end.

**Correction.** All documents now say 63 to 97 percent for the 14-model
study; the technical report keeps 65 to 98 only for the three prototype
models beside their table. Per-model values: Table "Misalignment
decomposition at each model's dip layer" in the ICLR appendix
(`paper/iclr2027/tables/budget_models.tex`).
