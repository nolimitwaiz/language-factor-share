# Pre-registration: code-switched training (proposal section 3.2.1)

**Status: FROZEN 2026-07-28, before any code-switch training arm was run.**
Author: Waiz Khan (wkhan12@jh.edu)

The corpus was generated on 2026-07-27 (`src/aim2/codeswitch.py`, 14
languages passing the alignment-agreement threshold, six configurations,
13,752 to 18,774 sentences each). No model has been trained on it. This
document fixes the design, the predictions and the decision rules first.

---

## 1. Why this is pre-registered more carefully than the last one

The LFS pilot froze seven criteria in advance and two of them turned out to
be incapable of returning a meaningful verdict:

- **D6**: criterion 2 required concept-variance preservation at or above 0.90
  *throughout training*. The control arm failed it. A criterion the control
  cannot pass cannot discriminate.
- **D7**: criterion 5 required context use to improve, with no noise band, on
  a measurement whose standard error was eight times the observed effects.
  Its verdicts were decided by the sign of noise.

Both errors share a cause: a rule was written by reasoning about the intended
direction of an effect without first asking what the quantity does under the
null. The corrections below are therefore procedural as much as numerical.
**Every criterion in section 5 states its minimum detectable effect, and any
criterion whose observed effect falls below that is reported as "not
detectable" rather than as a pass or a fail.**

---

## 2. Design

Base model Qwen3-0.6B-Base, 400 steps, three seeds per arm, matched token
budget and identical optimizer settings across arms. Training sentences come
from the code-switch corpus, whose range is 500 to 1900, disjoint from every
sentence any Aim 1 or Aim 2 number is reported on.

| arm | training text |
|:--|:--|
| CS-A | frozen model, no training (reference point only) |
| CS-B | **control**: the identical sentences with no substitutions applied |
| CS-W10 | word-level switching, rate 0.10 |
| CS-W25 | word-level switching, rate 0.25 |
| CS-W50 | word-level switching, rate 0.50 |
| CS-P25 | phrase-level switching, rate 0.25 |

CS-B is the control that matters. The corpus stores the unmodified source
sentence alongside every switched one, so the control sees the same
sentences, the same count and the same domain, differing *only* in whether
the substitutions were applied. Any difference is attributable to code
switching and to nothing else.

CS-P25 against CS-W25 isolates switch granularity at a matched rate.

### 2.1 A replay mixture, held constant across arms

Every arm draws 25 percent of each batch from unmodified multilingual text in
the switch languages. This is identical in every arm, so it cannot confound
the code-switch contrast.

It is included because the LFS pilot established that continued training on a
narrow language set degrades held-out languages by a geometric mean of 2.1 to
2.3 times *in every arm including the control*, which swamped the capability
criterion entirely and wasted roughly a third of that design. Without replay
this pilot would repeat that failure with certainty. Whether replay actually
prevents it is measured, not assumed, and is prediction P5.

### 2.2 Evaluation

The existing harness, unchanged: `src/aim2/evaluate.py` and
`src/aim2/context_use.py`. Evaluation uses the first 300 sentences, disjoint
from training, and reports separately on the switch languages and on eight
languages that appear in no arm's training data in any form.

---

## 3. Predictions, frozen

Directional, committed before running. P3 is the one worth the pilot.

**P1. Code-switched training improves mean-case retrieval on the switch
languages relative to CS-B.** This is the proposal's implicit hypothesis and
the reason the sub-aim exists. Predicted direction: MEXA up.

**P2. The effect is monotone in switch rate** across CS-W10, CS-W25, CS-W50.
Stated as a genuine risk: it is equally plausible that a high rate degrades
the text into something neither English nor the target language, in which
case the trend reverses at 0.50. Either outcome is informative and both are
recorded here in advance.

**P3. Tail alignment does NOT improve, and may degrade.** The LFS pilot found
that every intervention which drove LFS down also drove AaR down, Pearson
+0.924 across eighteen arm-seed points. That was observed for objectives that
act directly on representations. Code switching acts on the *data* and adds
no representational loss term at all. If AaR degrades here too, the effect is
general to multilinguality interventions at this scale rather than specific
to alignment losses. If AaR holds or improves, the earlier reversal is
specific to explicit alignment objectives, which would be the more useful
outcome for Aim 2. **This prediction is what distinguishes the two
explanations and neither result is a null.**

**P4. Concept variance is preserved in every arm, CVP at or above 0.90 on the
final checkpoint.** No arm carries an alignment or an LFS term, so nothing
pushes representations together directly and the collapse pathway should be
unreachable. If CVP falls anyway, collapse is reachable through data alone
and the guard's role is larger than currently believed.

**P5. The replay mixture reduces held-out degradation below the 2.1 to 2.3
times geometric mean measured without it.** If it does not, replay at 25
percent is insufficient and criterion 6 remains undiscriminating.

---

## 4. What would make this pilot uninformative

Stated in advance so it is recognizable.

- If CS-B and every switch arm land within noise of each other on every
  downstream measure, 400 steps is too short for data composition to matter
  at this scale, and the pilot speaks to the budget rather than to code
  switching.
- If the replay mixture drives all arms toward the frozen model, replay is
  dominating the update and the arms are not being compared at meaningful
  training pressure.
- If alignment quality rather than switch rate explains the ordering of arms,
  the manifest's per-language agreement figures are the confound, and the
  comparison must be restricted to languages above a common threshold.

---

## 5. Decision rules, with minimum detectable effects

An arm is promising only if all of the following hold in three of three
seeds. Each states what it can detect; anything smaller is reported as not
detectable rather than scored.

1. **Mean-case retrieval improves against CS-B** on the switch languages.
   Across-seed standard deviation of MEXA in the LFS pilot was 0.009, so the
   minimum detectable difference at two standard deviations is **0.018**.
2. **Concept variance preserved, on the final checkpoint**, CVP at or above
   0.90 against the frozen model. Evaluated at the checkpoint and not per
   step, per D6. The per-step floor is retained as a live monitoring tripwire
   only and returns no verdict.
3. **Trajectory criterion, replacing the per-step floor**: mean CVP over
   training at or above CS-B's mean minus one across-seed standard deviation.
   This is the discriminating form of criterion 2, scored against the control
   rather than against an absolute threshold, per D6.
4. **Tail alignment does not degrade** against CS-B by more than two
   across-seed standard deviations. Pilot standard deviation was 0.005, so
   the minimum detectable degradation is **0.010**.
5. **Cross-lingual context use**, measured at **200 sentence pairs** rather
   than 60. Per-language standard error was 0.047 at 60 pairs and scales as
   the inverse square root of the pair count, giving roughly **0.026** at
   200, so the minimum detectable difference at two standard errors is
   **0.052**. The differences observed in the LFS pilot were near 0.005 and
   would remain undetectable at any feasible pair count; if that is the true
   effect size here, this criterion is reported as **not detectable** and
   contributes no verdict. Recorded now so that outcome is not read as a
   failure to measure.
6. **Held-out capability**, with replay active: degradation no worse than
   CS-B's plus one percentage point of relative perplexity increase. This
   criterion is only meaningful if P5 holds; if replay fails to reduce
   forgetting, criterion 6 is reported as undiscriminating exactly as it was
   in the LFS pilot, and that is a result about the setup rather than about
   any arm.
7. **Direction consistent across all three seeds** for every criterion above
   that returns a verdict.

---

## 6. What this cannot establish

- One model, one scale, 400 steps. Nothing here generalizes to pretraining
  scale, where code-switched data would actually be deployed.
- The switch languages are the 14 that passed the alignment-agreement
  threshold, which is not a random sample of languages: it excludes exactly
  those whose alignments were unreliable, and those are disproportionately
  the low-resource languages the sub-aim most wants to help.
- Substitution quality is bounded by word alignment quality, which the
  manifest records per language and which varies from 0.82 to below the
  exclusion threshold. A null result may reflect alignment noise rather than
  the intervention.
- Code switching generated from parallel text is not natural code switching.
  It has the switch-point distribution of an alignment algorithm, not of a
  bilingual speaker, and no claim about naturalistic code switching follows.
