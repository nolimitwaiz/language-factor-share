# Aim 2 study scorecard, three seeds

Scored against the criteria frozen in `AIM2_STUDY_PLAN.md` section 7 before
any arm ran. Three seeds, six arms, downstream evaluation and criterion 5
measurement all complete.

## Verdict: every arm fails. No working objective was produced.

That is the honest headline and it is not a disappointing one. The study was
built to find out whether a metric can be optimized without being defeated,
and it answered: not by any of the five objectives tested. The strongest
result is a negative one that bears directly on Aim 1.

**The Aim 1 association reverses under intervention.** Across pretrained
models, Aim 1 observed that lower LFS at middle layers went with better
cross-lingual behaviour. Here, over all eighteen arm-seed points, LFS and
tail alignment move *together*: Pearson +0.924, Spearman +0.787. Every
objective that drove LFS down also drove AaR down. Driving the metric in the
direction the observational data called good made the measured behaviour
worse, in every arm, at every seed.

| arm | objective | LFS | AaR@10 | MEXA | CVP |
|:--|:--|--:|--:|--:|--:|
| A | frozen | 0.445 | -0.090 | 0.444 | 1.000 |
| B | LM only | 0.468 | -0.095 | 0.417 | 0.955 |
| C | LM + LFS | 0.361 | -0.138 | 0.430 | 0.886 |
| D | LM + align | 0.150 | -0.180 | **0.500** | **0.216** |
| E | LM + align + CVP | 0.178 | **-0.243** | 0.418 | 0.857 |
| F | all four | **0.111** | -0.238 | 0.476 | 0.942 |

Four trained languages, layer 8, mean of three seeds. Bold marks the extreme
of each column.

Three things follow, and the third is the one that cost the most to learn.

**MEXA and AaR dissociate completely under optimization.** Their rank
correlation across the eighteen points is -0.067, indistinguishable from
none. Arm D posts the best mean-case retrieval in the whole study, 0.500
against the frozen model's 0.444, while its concept variance collapses to
0.216 and its tail alignment falls well below frozen. An evaluation carrying
only mean-case retrieval would have reported arm D as the study's success.
The two instruments were built separately in Aim 1 on the argument that
mean-case retrieval hides tail behaviour; this is that argument being paid
out under direct optimization pressure.

**The concept-variance guard works and is not sufficient.** It was designed
to stop the collapse arm D exhibits, and it does exactly that: E and F hold
concept variance at 0.857 and 0.942 against D's 0.216. It does not rescue the
outcome. E and F post the *worst* tail alignment of any arm, -0.243 and
-0.238 against frozen -0.090. Preserving the variance the metric is defined
over does not preserve the behaviour the metric was meant to proxy for. The
guard closes the specific loophole it was built for and the objective finds
another.

**Continued training on four languages degrades the other eight, whatever the
objective.** Held-out perplexity rises by a geometric mean of 2.10 to 2.31
times across every arm including B, which carries no alignment or LFS term at
all. Yoruba worsens by 6.5 to 8.7 times and Zulu by 7.0 to 7.9. This is a
property of the continued-training setup rather than of anything under test,
and it means no arm could have passed criterion 6 regardless of its
objective. Any future version of this study needs a replay or rehearsal
mixture before its capability criterion can discriminate between arms.

Model Qwen3-0.6B-Base, 400 steps, alignment pressure at layer 8 (the Aim 1
dip layer), four languages, eight concepts per batch, sentences 500 to 1900,
disjoint from every sentence any Aim 1 number is reported on.

---

## 1. A correction to how the batch quantity was described

An earlier note in this project said the first study run was invalid because
its batch LFS of 0.585 fell "below its own noise floor" of 0.700. That
description was wrong and is corrected here.

The pure-noise value `(L-1)/((L-1)+(K-1))` is what LFS equals when there is
neither language nor concept structure present. It is a reference point, not
a bound. A value **above** it means language structure dominates; a value
**below** it means concept structure dominates. A batch LFS beneath the
pure-noise value is therefore meaningful evidence of genuine cross-language
alignment, not a broken measurement.

The real defect in the eight-language configuration was different, and is
about gradient quality rather than validity. The objective pushes down
`V_language`, which is computed from the L language means. At L=8 and K=4
each of those eight means was estimated from only four concept samples, so
the exact term the objective differentiates was dominated by sampling noise.
Reshaping to L=4 and K=8, at an identical thirty-two sequences per batch,
estimates each of the four language means from eight samples instead. The
pure-noise value falling from 0.700 to 0.300 is a *summary* of that
improvement in degrees of freedom, not the reason for the change.

The batch pure-noise value is now printed at startup and stored in every
result file, so no loss curve from this study can be read without it.

---

## 2a. Replication across three seeds

Mean CVP over training, per arm per seed. Every arm ran 400 steps at the same
configuration, differing only in seed.

| arm | objective | seed 0 | seed 1 | seed 2 | mean | sd | floor breaches |
|:--|:--|--:|--:|--:|--:|--:|:--|
| B | LM only | 0.985 | 0.969 | 0.974 | 0.976 | 0.008 | 2, 1, 2 |
| C | LM + LFS | 0.878 | 0.930 | 0.797 | 0.868 | 0.067 | 12, 10, 15 |
| D | LM + align | 0.575 | 0.573 | 0.546 | **0.565** | 0.016 | 16, 15, 17 |
| E | LM + align + CVP | 0.908 | 0.896 | 0.925 | 0.910 | 0.014 | 8, 11, 10 |
| F | all four | 1.016 | 0.971 | 0.939 | 0.975 | 0.039 | 8, 11, 12 |

**The alignment-harm result replicates at three of three seeds.** Arm D sits
below arm C by 0.303, 0.357 and 0.251 in the three seeds, never once
reversing. D is also the tightest arm across seeds, standard deviation 0.016,
so the effect is not a matter of one unlucky run. The alignment term alone is
reliably the most destructive term in this study, more so than the naive LFS
objective it was included as a foil for.

**The guard result replicates at three of three seeds, and is stronger than
seed 0 suggested.** E exceeds D by 0.333, 0.323 and 0.379. The
language-modelling cost is not merely absent: E's final LM loss is *lower*
than D's in all three seeds, by 0.002, 0.015 and 0.018. Adding a constraint
that preserves concept variance did not trade against language modelling
here, it accompanied a small improvement in it. With three seeds and
differences this small the honest reading is that the guard is free rather
than that it helps, but the direction is consistent and no seed shows a cost.

**Criterion 2 is failed by the control at every seed**, breaching 2, 1 and 2
times. This confirms D6 in the discrepancy ledger as a property of the
criterion rather than an artifact of one run.

### A caveat that the three seeds exposed

Batch LFS at step 0 was 0.282, 0.635 and 0.498 in the three seeds. The model
is identical and untrained at that point, so the entire spread comes from
which concepts and languages happened to land in the first batch. The
across-seed standard deviation at step 0 is 0.178, which is larger than most
of the training-time movements being discussed.

Any within-seed comparison of batch LFS from start to end is therefore
dominated by batch sampling noise and should not be read as an effect size.
The end-of-training values are far tighter, 0.193, 0.175 and 0.252, standard
deviation 0.040, but that convergence is itself a result rather than a
licence to treat the start values as a baseline. This is precisely why
criterion 1 is scored on the downstream evaluation, which uses a fixed set of
300 sentences and all concepts, rather than on the training log.

---

## 2b. What the training dynamics show, seed 0 in detail

| arm | objective | LFS start | LFS end | CVP mean | CVP min | breaches | LM end |
|:--|:--|--:|--:|--:|--:|--:|--:|
| A | frozen | — | — | — | — | — | — |
| B | LM only | 0.282 | 0.193 | 0.985 | 0.855 | 2/21 | 2.006 |
| C | LM + LFS | 0.282 | 0.013 | 0.878 | 0.449 | 12/21 | 2.081 |
| D | LM + align | 0.282 | 0.082 | 0.575 | 0.287 | 16/21 | 2.180 |
| E | LM + align + CVP | 0.282 | 0.153 | 0.908 | 0.691 | 8/21 | 2.177 |
| F | LM + align + CVP + LFS | 0.282 | 0.008 | 1.016 | 0.647 | 8/21 | 2.242 |

Batch pure-noise value 0.300. CVP floor 0.90. Breaches count logged steps
with CVP below the floor, out of twenty-one.

**Finding 1, the intended demonstration succeeded.** Arm C drove batch LFS
from 0.282 to 0.013, a ninety-five percent reduction, while concept variance
fell to 0.449 of the frozen model's. Reported alone, that LFS number would
read as a large success. It is the metric being defeated rather than
improved, and the guard instrument caught it while it happened rather than
in hindsight. This is the contrast the study was built to produce.

**Finding 2, and this one was not predicted.** The alignment term on its own,
arm D, damaged concept variance *more* than the naive LFS objective did:
mean CVP 0.575 against C's 0.878, minimum 0.287 against 0.449, and sixteen
floor breaches against twelve. Pushing translations of the same sentence
together is the standard move in the cross-lingual alignment literature, and
in this study it was the single most destructive term tested. If this holds
across seeds it is the most useful result here, because it is a caution about
a widely used technique rather than about a metric this project invented.

**Finding 3, the guard term does measurable work.** D and E differ only by
the CVP hinge. E holds mean CVP 0.333 higher than D, at a language-modelling
loss that is indistinguishable, 2.177 against 2.180. That is a clean
attribution: the guard recovers a third of the lost concept variance at no
measured cost to the language-modelling objective.

**Finding 4, the guard is under-weighted at the value chosen.** E still
breached the floor eight times out of twenty-one and reached 0.691 at its
worst. The hinge is quadratic in the violation, so a shortfall of 0.05
produces a penalty near 0.0025 against an alignment loss of order 3.4. The
weight is too small to enforce the constraint it names. A weight sweep is
needed before E or F is described as protecting anything.

**Finding 5, section 8's first warning fired.** Arm B moved batch LFS from
0.282 to 0.193 carrying no LFS term whatsoever. Plan section 8 stated in
advance that if ordinary continued training moves LFS substantially, the
metric is responding partly to data domain, and every comparison must be
made against B rather than against the frozen model. It does, and they are.
The downstream evaluation computes every contrast against B for this reason.

---

## 3. Scoring against the frozen criteria

An arm is promising only if all seven hold, in three of three seeds.

| # | criterion | B | C | D | E | F |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | LFS falls at target layer vs B | ref | pass | pass | pass | pass |
| 2 | CVP at or above 0.90 throughout | **fail** | **fail** | **fail** | **fail** | **fail** |
| 3 | residual share rises at most 5 pts | pass | pass | pass | pass | pass |
| 4 | MEXA or AaR improves vs B or holds | ref | fail | pass | mixed | pass |
| 5 | context use improves vs B | ref | *noise* | *noise* | *noise* | *noise* |
| 6 | capability degrades at most B + 1 pt | ref | **fail** | **fail** | **fail** | **fail** |
| 7 | direction consistent across seeds | — | fail | fail | fail | fail |

**Every arm fails.** Two criteria do the work, and each fails for a reason
worth separating from the objectives under test.

Criterion 2 is failed by every arm *including the control*, which is D6 in
the discrepancy ledger: a criterion the control cannot pass cannot
discriminate. Criterion 6 is failed by every arm because continued training
on four languages degrades the other eight by more than two-fold regardless
of objective, which is a property of the setup. Neither failure is evidence
about the objectives, and saying that plainly matters more than the tally.

Criterion 5 returned verdicts it had no power to support. The arm differences
run from -0.006 to +0.006 against a per-language standard error of 0.047,
roughly eight times larger, so the passes and failures are decided by the
sign of noise and flip between seeds for that reason. Recorded as D7. The
verdicts are left exactly as the frozen rule produced them rather than
recomputed under a rule invented after seeing the data; at 60 sentence pairs
this design can detect a difference of about 0.10 and no smaller.

**What the criteria that did work say.** Criterion 1 passes everywhere, which
is the whole lesson: LFS fell in all five arms and not one of them improved
the behaviour it was supposed to track. Criterion 4 passes for D and F on
mean-case retrieval alone while both post tail alignment well below the
frozen model, which is the MEXA and AaR dissociation restated in the
scoring's own terms.

---

## 4. What this does not show

- **One model, one layer, one scale.** Qwen3-0.6B-Base, layer 8, 400 steps,
  four languages. The reversal of the Aim 1 association is demonstrated here
  and nowhere else; whether it survives at larger scale, at other layers, or
  under longer training is untested and the study cannot speak to it.
- **The intervention may be too blunt rather than the association wrong.**
  An alternative reading of the reversal is that these five objectives are
  crude instruments and a better-designed one would move LFS without the
  collateral damage. Nothing here rules that out. What the study establishes
  is that the association does not survive *these* interventions, which is
  weaker than the claim that it cannot survive any.
- **Criterion 6 could not discriminate.** Without a replay mixture, held-out
  language degradation swamps any difference between objectives, so the
  capability comparison the plan asked for was not actually run.
- **Criterion 5 was underpowered by roughly a factor of eight** and its
  verdicts carry no information. Raising the pair count is cheap and is the
  first thing to fix.
- **Tail alignment is measured by AaR@10 at 300 sentences**, where the tail
  is thirty sentences. The dissociation between AaR and MEXA is large and
  consistent across eighteen points, but the tail estimate itself is coarse.
- **No claim is made that alignment objectives are harmful in general.** Arm
  D was the most destructive term tested in this configuration. Whether that
  is a fact about alignment objectives or about this learning rate, this
  layer, and this batch shape is exactly the kind of question the study was
  meant to raise rather than settle.

---

## 5. What to change before running this again

In order of expected value.

1. **Add a replay mixture.** Until held-out degradation is controlled,
   criterion 6 cannot separate arms and roughly a third of the design is
   wasted.
2. **Re-specify criteria 2 and 5** per D6 and D7, freezing the replacements
   with a power calculation attached, before any further run.
3. **Sweep the guard weight.** The hinge at its current weight produces a
   penalty near 0.0025 against an alignment loss of order 3.4. The guard's
   effect on concept variance is large and consistent already; what an
   adequately weighted version does to tail alignment is unknown and is the
   most direct follow-up.
4. **Add an arm that optimizes AaR directly** and check whether LFS follows.
   The study tested one direction of the association and found it does not
   hold; the other direction is untested and cheap.
5. **Report MEXA and AaR jointly, never MEXA alone.** Their correlation under
   optimization is -0.067. Any future arm reported on mean-case retrieval by
   itself would be reported misleadingly, and arm D is the worked example.
