# Pre-registration: word-level alignment objectives (proposal section 3.2.2)

**Status: FROZEN 2026-07-28, before any WA arm was run at full length.**
Author: Waiz Khan (wkhan12@jh.edu)

A four-step smoke run was executed first to confirm the harness loads
alignments and computes the loss. Its numbers appear below as the basis for
the weighting choice, and no downstream quantity was measured from it.

---

## 1. Why this exists: the earlier study did not test 3.2.2

Section 3.2.2 specifies, as its primary form, a loss on **word**
representations linked by a bilingual dictionary, and names a variant in
which **a language identity vector is subtracted before the difference is
computed**. Only then does it add that segment-level pooled variants will
*also* be explored.

The six-arm study ran mean-pooled segment representations with an InfoNCE
contrastive loss. That deviates from the section in three separate ways:

1. **Segment level, not word level.** It ran the "in addition" variant.
2. **Contrastive, not difference.** The section says the loss "measures the
   difference between word representations for the matched words." InfoNCE is
   a contrastive objective with negatives; a difference loss has none. These
   are different objective families with different failure modes.
3. **The language identity vector was never implemented**, so the section's
   own named variant is untested.

Its conclusion, that every arm failed, is therefore a valid result about a
segment-level contrastive objective and **is not a result about 3.2.2**. That
correction is the reason for this study.

---

## 2. The specified objective has a degenerate minimum, and that is the point

A mean squared difference between matched word representations is globally
minimized at `h = 0`. Shrinking every representation toward zero drives the
term to nothing while destroying the content the representations carry.

The smoke run makes the risk concrete rather than theoretical: at
initialization the word term is **615 to 2488** against a language modelling
loss near **10**. Unweighted, the specified objective outweighs language
modelling by two orders of magnitude and the degenerate solution is the
cheapest path available to the optimizer.

`WORD_W = 0.02` balances the two terms at initialization. **This is a choice
the proposal does not make**, and it is recorded here rather than presented as
neutral. It slows the degeneracy; it does not remove it.

**This is the clearest test yet of whether Aim 1's instrument is needed for
Aim 2's method.** If the specified objective drifts toward the degenerate
solution and concept-variance preservation detects it while retrieval metrics
do not, then the metric suite is not decoration on the training work: it is
what stands between the proposal's own objective and a silent failure.

---

## 3. Design

Qwen3-0.6B-Base, 400 steps, three seeds, six languages with two-aligner
agreement from 0.61 to 0.75 (deu 0.750, fra 0.748, ces 0.737, swa 0.691,
hin 0.676, ben 0.606), 8334 aligned sentence pairs drawn from sentences 500
to 1900, disjoint from all evaluation data.

| arm | objective |
|:--|:--|
| WA-A | frozen reference |
| WA-B | **control**: language modelling only, identical data and steps |
| WA-C | + word-level squared difference on linked words (the section's primary form) |
| WA-D | WA-C + learned language-identity vector subtracted (the section's variant) |
| WA-E | + segment-level squared difference (the "in addition" form) |
| WA-F | WA-D + concept-variance guard |
| WA-G | WA-D with cosine distance instead of squared difference |

The contrasts each answer one question:

- **WA-C vs WA-B**: does the specified objective help at all?
- **WA-D vs WA-C**: does the language-identity vector help? *This is the
  proposal's own hypothesis and the most direct Aim 1 to Aim 2 link
  available, because subtracting a per-language vector is removing exactly
  the component LFS measures.*
- **WA-E vs WA-C**: word level against segment level, matched on everything
  else. The earlier study could not make this comparison.
- **WA-F vs WA-D**: does the guard prevent the degeneracy?
- **WA-G vs WA-D**: does a scale-invariant distance avoid it by construction?

---

## 4. Predictions, frozen

**P1. WA-C drifts toward the degenerate solution.** Mean representation norm
at the monitored layer falls by more than 25 percent against WA-B, and
concept-variance preservation falls below 0.90 on the final checkpoint.
Logged as `rep_norm` every 20 steps so the drift is visible while it happens.

**P2. Retrieval will not reveal it.** MEXA on the switch languages holds or
improves in WA-C even as the norm falls, because retrieval is computed on
cosine similarity and is invariant to scale. **If P1 holds and P2 holds, the
specified objective fails in a way the field's standard metric cannot see.**

**P3. The language-identity vector helps.** WA-D beats WA-C on tail alignment.
Reasoning: the vector gives the model an explicit place to put
language-specific information, so satisfying the alignment term no longer
requires removing that information from the representation itself. This is
the proposal's stated intuition and it has never been tested.

**P4. WA-G does not degenerate.** Cosine distance is scale-invariant and has
no `h = 0` minimum, so its representation norm stays within 10 percent of
WA-B's.

**P5. Word level beats segment level.** WA-C exceeds WA-E on tail alignment,
because a segment mean can be matched while the individual words underneath
it are misaligned, and the tail is where that slack shows.

Stated risk on P5: it is equally plausible that word-level matching is
noisier, since it inherits alignment error at 0.61 to 0.75 agreement, and
that the segment mean averages that noise away. Either outcome is recorded
here in advance.

---

## 5. Decision rules with minimum detectable effects

Same discipline as `PREREG_CODESWITCH.md`. An effect below its MDE is
reported **not detectable** and returns no verdict.

1. **Representation norm** against WA-B. Across-seed standard deviation is
   unknown for this quantity, so it is estimated from the three seeds and
   reported; a drift is called only above two estimated standard deviations,
   and the estimate is published alongside.
2. **Concept variance on the final checkpoint**, at or above 0.90 against
   frozen, per D6. Per-step floor is a monitoring tripwire only.
3. **Mean-case retrieval** against WA-B. MDE **0.018**, from the LFS study's
   across-seed standard deviation of 0.009.
4. **Tail alignment** against WA-B. MDE **0.010**, from an across-seed
   standard deviation of 0.005.
5. **Capability**, held-out perplexity no worse than WA-B plus one
   percentage point.
6. Direction consistent across all three seeds for anything returning a
   verdict.

---

## 6. What this cannot establish

- 400 steps at 0.6B, six languages, one layer.
- Alignment links are bounded by aligner agreement of 0.61 to 0.75, so a null
  on the word-level arms may reflect link noise rather than the objective.
  Languages below 0.50 agreement are excluded, which biases the sample toward
  languages the aligners handle well.
- The weighting choice is ours, not the proposal's. A different balance
  between the language modelling and alignment terms could change every
  ordering here, and no sweep over it is run.
- `WORD_W` was set from a smoke run rather than from a principled criterion.
  It equalizes the two terms at step 0 and says nothing about their relative
  importance thereafter.
