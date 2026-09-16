# Code-switch study scorecard, three seeds

Scored against `prereg/PREREG_CODESWITCH.md` section 5, frozen before any arm
ran. Six arms, three seeds, Qwen3-0.6B-Base, 400 steps, 25 percent replay
mixture identical in every arm. Evaluation on the first 300 sentences,
disjoint from the training range.

## Verdict: no detectable benefit, and a large measurable cost

Every point estimate on every downstream measure is negative or zero, and all
but one fall below their pre-registered minimum detectable effect. The one
that clears it clears it in the wrong direction. Meanwhile the cost is neither
small nor ambiguous: code-switched training degrades languages outside the
replay set by 17 to 47 percentage points of perplexity relative to the
control, monotonically in switch rate.

| criterion | CS-W10 | CS-W25 | CS-W50 | CS-P25 |
|:--|:--|:--|:--|:--|
| 1. mean retrieval vs control (MDE 0.018) | $-0.012$ nd | $-0.011$ nd | $-0.000$ nd | $-0.005$ nd |
| 2. concept variance, final checkpoint | pass | pass | **FAIL** | pass |
| 3. trajectory vs control | pass | pass | pass | pass |
| 4. tail alignment (MDE 0.010) | $-0.006$ nd | **$-0.010$ FAIL** | $-0.005$ nd | $-0.005$ nd |
| 5. context use (MDE 0.052) | $-0.006$ nd | $-0.028$ nd | $-0.039$ nd | $-0.011$ nd |
| 6. capability vs control + 1 pt | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

"nd" is **not detectable**: the observed effect is smaller than the minimum
detectable effect frozen for that criterion, so it returns no verdict. That
outcome was pre-registered as a possible result precisely so it could not be
read later as a pass or a fail.

**P1 is not supported.** Code-switched training did not improve mean-case
retrieval on the switch languages. This was the proposal's implicit
hypothesis and the reason the sub-aim exists. The honest statement is not
"code switching fails" but "at this scale, with this budget, on this corpus,
no benefit is detectable and every point estimate is slightly negative."

**P2 is not supported.** The effect is not monotone in switch rate. The three
word-level rates give $-0.012$, $-0.011$ and $-0.000$, which is no trend at
all rather than the reversal-at-high-rate the prereg named as the alternative.

---

## The result worth the study: P3

**The LFS and tail-alignment coupling is largely specific to objectives that
act on representations.**

| study | intervention acts on | Pearson | n | p |
|:--|:--|--:|--:|--:|
| LFS study | representations (alignment, LFS, guard terms) | $+0.924$ | 18 | $<10^{-7}$ |
| this study | data only (code-switched text) | $+0.419$ | 15 | 0.120 |

The LFS study found that every objective driving LFS down also drove tail
alignment down, and reported that as the observational Aim 1 association
failing under intervention. This study narrows that considerably. When the
intervention changes only the training data and adds no representational loss
term, the coupling is not detectable at conventional significance.

The narrowing matters for how the Aim 1 result should be stated. The reversal
is not a general property of doing something to a model's multilinguality; it
is a property of pushing directly on the representation geometry the metric
is computed from. That is a more precise and more useful claim than the one
the LFS study alone supported, and it is the reason P3 was written as the
prediction that would distinguish the two explanations.

Stated conservatively: $+0.419$ at $n{=}15$ is not evidence of no coupling.
It is failure to detect one, in a design with three arms' worth of spread in
LFS rather than the wide spread the LFS study's objectives produced.

---

## The one clear signal: replay protects what it covers, and nothing else

Perplexity relative to the control arm, by language group. Groups are derived
from the corpus manifest per D8, not inherited from the previous study.

| group | languages | CS-W10 | CS-W25 | CS-W50 | CS-P25 |
|:--|:--|--:|--:|--:|--:|
| switch-exposed | deu, hin, swa | 0.994 | 1.004 | 0.983 | 0.999 |
| replay-exposed | fra, rus, zho-CN, ben, ind | 1.014 | 1.041 | 1.040 | 1.039 |
| **genuinely unseen** | **arb, yor, zul** | **1.126** | **1.215** | **1.360** | **1.237** |

The genuinely unseen languages degrade **monotonically in switch rate**,
1.126 to 1.215 to 1.360 across the three word-level arms, while the languages
the replay mixture covers stay within four percent of the control. This is
the only monotone, interpretable effect in the study.

The reading is straightforward. Replay protects the languages it contains and
does nothing for the languages it does not. Code-switched text displaces
ordinary text in the batch, and the cost of that displacement falls entirely
on languages outside the rehearsal set, in proportion to how much was
displaced. A 50 percent switch rate costs 36 percent perplexity on unseen
languages for no measurable gain anywhere else.

**Caveat carried from D8: this group is three languages.** The monotonicity
is clean and the effect is large, but it rests on Arabic, Yoruba and Zulu
alone, because five of the eight languages the pre-registration called
held-out are in the corpus and received full-sentence replay. That error cost
this comparison most of its power and is the single biggest methodological
loss in the study.

---

## P5 holds, and it rescued criterion 6

Perplexity against the **frozen** model, geometric mean by group.

| group | CS-B (control) | CS-W10 | CS-W25 | CS-W50 | CS-P25 |
|:--|--:|--:|--:|--:|--:|
| switch-exposed | 0.730 | 0.725 | 0.732 | 0.715 | 0.729 |
| replay-exposed | 0.965 | 0.979 | 1.005 | 1.003 | 1.002 |
| genuinely unseen | 1.415 | 1.584 | 1.701 | **1.887** | 1.723 |

**P5 is supported.** The LFS study, with no replay, degraded held-out
languages by 2.10 to 2.31 times in every arm including its control. Here the
control sits at 1.415 and even the worst arm at 1.887, both below that range.
A 25 percent replay mixture measurably reduces forgetting.

**The consequence is the point.** In the LFS study criterion 6 was
undiscriminating: every arm including the control degraded by roughly the
same large factor, so the criterion could not separate objectives and a third
of that design was wasted. Here it separates them cleanly and monotonically:

| arm | genuinely unseen, vs control | verdict |
|:--|--:|:--|
| CS-W10 | $+16.98$ pts | FAIL |
| CS-W25 | $+28.68$ pts | FAIL |
| CS-W50 | $+47.26$ pts | FAIL |
| CS-P25 | $+30.81$ pts | FAIL |

Every switch arm fails criterion 6, and this time the failure is about the
intervention rather than about the setup. That distinction is what the replay
mixture bought, and it is the clearest methodological payoff in either study:
a fix identified as the top-ranked follow-up from the LFS study was applied,
and it converted a dead criterion into the one that carries the result.

Note also that switch-exposed languages *improve* against the frozen model,
to 0.730, in the control as much as in any switch arm. That gain comes from
the replay mixture and from ordinary continued training on those languages,
not from code switching, which is exactly why every criterion is scored
against CS-B rather than against the frozen model.

---

## The confound I flagged does not explain the results

Final language-modelling loss rises monotonically with switch rate, 1.140 for
the control up to 2.589 at a 50 percent rate, because code-switched text is
genuinely harder to model. That was raised in advance as a possible driver of
any downstream ordering: an arm that fit its own data less well might look
worse for reasons unrelated to code switching.

It does not hold up. Across the five arms, final LM loss ranks against mean
retrieval at Spearman $-0.100$ and against tail alignment at $-0.500$, on
five points. Neither is meaningful, and the downstream measures are flat
across arms whose LM losses differ by more than a factor of two. The
confound was real to worry about and is not what produced these numbers.

---

## Two errors found during this study, both the same kind

**D8**, found after training and before any result was read: the
pre-registration claimed eight evaluation languages appeared in no arm's
training data, and five of them are in the corpus and in the replay mixture.
The frozen text stands as written and wrong with the ledger entry attached;
the analysis reports three groups instead of two.

**The frozen-reference bug**, found while the evaluation ran: both evaluation
scripts identified the untrained model by the literal arm name `A`. This
study names it `CS-A`, so it fell through to the checkpoint branch, found
none, and skipped silently. Criteria 1, 4 and 5 are unaffected because they
contrast against CS-B, which is the control the pre-registration specifies.
Criterion 6 and P5 lost their baseline entirely and required a backfill run.

Both are the same failure: a value correct in one study carried into another
without re-checking that it still applied. With D4 and D6 that makes four
instances in this project. The procedural fix already recorded, deriving
groupings from the manifest at scoring time, addresses the data half; the
code half is addressed by identifying the frozen reference by set membership
rather than by a literal.

---

## A pre-registered number that came in wrong

The prereg computed the context-use standard error at 200 pairs as 0.026, by
scaling the LFS study's 0.047 by the inverse square root of the pair count.
The observed value is **0.0333**, so the true minimum detectable effect is
**0.067 rather than the 0.052 frozen in the document**.

The scaling law held; the base did not, because 0.047 was an average across
that study's languages and the six used here are noisier. No verdict changes,
since every criterion 5 effect falls below even the understated threshold.
Recorded because a power calculation that is quietly wrong is worse than none
when the next study reuses it.

---

## What this does not establish

- 400 steps at 0.6B. Code-switched data would be deployed at pretraining
  scale, where the displacement cost this study measures may be negligible
  and the benefit may appear. Nothing here speaks to that.
- The 14 corpus languages are those that passed the alignment-agreement
  threshold, which excludes precisely the languages whose alignments were
  unreliable, and those are disproportionately the low-resource languages the
  sub-aim most wants to help.
- Substitution quality is bounded by word alignment quality. A null result
  may reflect alignment noise rather than the intervention.
- This is not naturalistic code switching. Switch points come from an
  alignment algorithm, not a bilingual speaker.
- Naturalistic code switching, larger scale, and corpora covering the
  low-resource languages excluded here all remain untested.

---

## A note on terminology

These runs are called *studies* in the analysis documents and in the technical
report. The pre-registration files use the older word "pilot" throughout, and
they are left exactly as frozen. A frozen document that gets edited later,
even for a word with no bearing on any decision rule, is no longer evidence
that the rules predated the results. The wording differs; nothing else does.
