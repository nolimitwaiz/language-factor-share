# PRE-REGISTRATION: Word-level Language-Factor Share

**Frozen 2026-07-27, before any word-level representation has been
extracted.** Stage 0 (alignment infrastructure) is complete and its results
are inputs to this document; nothing downstream of alignment has been run.
Author: Waiz Khan.

---

## 0. What Stage 0 established, and how it constrains this design

Stage 0 measured alignment reliability on 20 languages with two
methodologically independent aligners. Two results bind this design.

**The Stage 0 gate failed.** Inter-aligner agreement above 0.70 F1 was
required on high-resource pairs; 2 of 7 cleared it (mean 0.636). The
response adopted here is not to trust one aligner but to **use only links
both aligners produce and that are one-to-one on each side**. Reliability is
bought with coverage: mean content-word coverage falls from 0.65 to 0.45,
and the per-language cost ranges from 16 percent of links retained (Burmese)
to 84 percent (Indonesian). Coverage is reported beside every number.

**The complete-grid requirement scales badly in the number of languages.**
LFS requires every concept to be present in every language. At word level
the concept set is the intersection of English positions aligned in *all*
selected languages, and that intersection decays multiplicatively:

| Languages | Shared concepts | Estimator pure-noise value |
|---|---|---|
| 4 | 1336 | 0.002 |
| 11 | 255 | 0.038 |
| 14 | 125 | 0.095 |
| 16 | 47 | unusable |

This is a structural property of word-level multilingual analysis, not a
compute limitation, and it does not arise at sentence level where
parallelism is given by the corpus rather than established by an aligner.
**It is reported as a finding in its own right**, independent of whether the
measurements below succeed.

---

## 1. Design, frozen

**Concept unit.** A specific English word occurrence, that is the
(sentence index, English token index) pair, restricted to content words as
defined on the English side. This keeps the grid structurally identical to
sentence level so the existing decomposition applies unchanged with a new
concept unit.

**Link set.** Agreed one-to-one: produced by SimAlign's intersection method
AND by eflomal, and one-to-one on both sides.

**Two operating points, both reported, neither chosen after seeing results.**
- **Primary, breadth:** the 11 languages with agreement F1 at least 0.60
  (spa, ind, deu, khm, fra, vie, ces, swa, hin, pol, zho-CN), N = 255
  shared concepts.
- **Secondary, precision:** the 4 languages with agreement at least 0.70
  (spa, ind, deu, khm), N = 1336 shared concepts.
The primary is the headline. The secondary exists because it has five times
the concepts and therefore much better per-cell precision; if the two
disagree, that disagreement is the result.

**Layers.** The model's LFS-dip layer and its two neighbours on each side,
read from existing grid results, never recomputed.

**Extraction order.** Alignment first, then extraction. Hidden states are
retained only at aligned positions and only at the five selected layers.
Dumping all token states and aligning afterward is not attempted; the
storage does not survive it.

**Subword pooling.** A word maps to a variable number of subwords per
language. Primary is the mean over a word's subwords, in fp32. First-subword
and last-subword are computed as variants. This is the same artifact as
sentence-level pooling and is worse here because the counts are small, so
all three are reported.

**Model.** Qwen3-1.7B for the prototype, which has campaign representations
and a known dip layer.

---

## 2. Gates. All must pass before any word-level number is reported as a result.

**G-WL.1 Reconstruction.** Averaging the aligned word vectors within a
sentence must approximately recover a sentence-level LFS computed **on the
same aligned content words only**, not on standard sentence pooling.
Standard pooling also includes function words, punctuation, and unaligned
tokens, so comparing against it would fail for a benign reason. Threshold:
Pearson correlation above 0.80 across the selected layers. Failure means the
pipeline is wrong and everything downstream is noise.

**G-WL.2 Monolingual null.** One language's aligned words split into
disjoint groups relabelled as pseudo-languages must sit at the estimator's
pure-noise value `(L-1)/((L-1)+(N-1))` to within 0.02, matching what was
established at sentence level.

**G-WL.3 Permutation null.** With alignments shuffled within sentence pairs,
word-level LFS must fall to the permutation null, and the real value must
exceed the 95th percentile of that null.

---

## 3. Confound controls, all required, not optional

- **Coverage equalization.** Aligned words are subsampled per language to
  the cross-language minimum, so per-cell precision does not vary with
  coverage. Without this a well-aligned language contributes cleaner
  estimates and appears less language-specific for a purely statistical
  reason.
- **Alignment confidence.** Results stratified into terciles of confidence;
  the conclusion must hold within each.
- **Word frequency.** Frequent words act as hubs and frequency
  distributions differ across languages; results stratified into frequency
  terciles.
- **Token position.** Controlled by matched-band subsampling, since position
  in the sequence affects representations in a causal model.

---

## 4. Directional predictions, frozen

**P-WL.1** Word-level LFS at the dip layer **exceeds** sentence-level LFS
computed on the same languages and sentences. Reasoning: averaging over a
sentence pools away per-word language marking, while a single word retains
morphology and script information. A word is expected to be more
language-identifiable than the sentence containing it.

**P-WL.2** The U-shaped depth profile persists at word level: the minimum
lies strictly between the first and last selected layer.

**P-WL.3** Word-level and sentence-level LFS agree on the ordering of
languages by language-specificity, Spearman above 0.5. A failure would mean
the two units measure different things about the same languages, which
would be a substantive finding.

**P-WL.4** The primary (11 language) and secondary (4 language) operating
points agree in direction on P-WL.1 and P-WL.2.

**P-WL.5** Subword pooling choice changes the absolute value but not the
sign of P-WL.1 nor the location of the minimum by more than one layer.

Scored cell by cell. Failures reported at the same prominence as passes.

---

## 5. Stage 2, conditional on Stage 1 passing its gates

Concept unit changes from word occurrence to **word type**, which splits
variance three ways: which word, which language, which context.

```
wLFS       = V_language / (V_language + V_word)
SenseShare = V_context  / (V_word     + V_context)
```

SenseShare is the quantity sentence-level LFS structurally cannot express:
how much of a word's representation is context-dependent rather than
identity-fixed. Restricted to word types with at least 10 occurrences so the
context factor is estimable. Given that the primary operating point has 255
concepts total, **it is likely that Stage 2 will require the 4-language
secondary point to have enough repeated types**, and if so that restriction
is reported rather than worked around.

Stage 2 predictions will be added as a dated addendum before it runs, not
now, because the type inventory is not yet known.

---

## 6. What would make this uninformative

Stated in advance so it is recognizable:

- If coverage equalization reduces the per-language sample below roughly 100
  words, per-cell estimates are too noisy to interpret and the analysis
  stops at the reported coverage table.
- If the reconstruction gate fails, no word-level number is reported at all.
- If word-level LFS sits at the pure-noise value for the chosen (L, N), the
  measurement has no signal above the estimator's floor and says nothing
  about language structure.

---

## ADDENDUM W1 (dated 2026-07-27, before the affected run)

### Corpus size raised from 300 sentences to the full NTREX (1997)

**Why.** Stage 0 aligned 300 sentences, a figure inherited from the
sentence-level convention. That was a design error specific to word level.
At sentence level, 300 sentences means 300 concepts present in every
language, guaranteed by the corpus. At word level the concept set is the
*intersection* of English words aligned in every language, which shrinks
multiplicatively, so 300 sentences yielded only 154 shared content concepts
across 11 languages. Because the estimator's pure-noise value is
`(L-1)/((L-1)+(N-1))`, adding languages raises the numerator while shrinking
N, and at 11 languages the floor (0.067) overtook the measured value
(0.063). The declared primary operating point became uninformative for a
reason that is entirely a consequence of corpus size, not of the phenomenon.

**Change.** Alignment and Stage 1 are re-run on all 1997 NTREX sentences.
Projected concept counts and floors:

| Languages | Content concepts | Pure-noise value |
|---|---|---|
| 4 | ~5700 | 0.0007 |
| 11 | ~1030 | 0.011 |
| 14 | ~520 | 0.026 |

**What is NOT changed.** No prediction is added, removed, or reworded. No
threshold is moved. No language is added to or removed from either declared
operating point. The gates are identical.

**Why this is a power increase and not a post-hoc adjustment.** The
pre-registration named insufficient N as a condition that would make the
measurement uninformative, and that condition occurred. The response is to
supply more data under the same design, not to relax a criterion that was
missed. The one prediction that failed on the small sample, P-WL.1, failed
for a reason established by a separate control (averaging inflates the
language share on structureless synthetic data), and that reason is
independent of N, so a larger sample is not expected to reverse it. If
P-WL.1 passes at the larger sample, that reversal is itself reportable and
would require explaining why the averaging control does not apply.

**A third operating point is added for reporting only, not as a headline:**
14 languages at agreement F1 above 0.55, marginal at a floor of 0.026, to
show how far breadth can be pushed before the floor overtakes the signal
again.
