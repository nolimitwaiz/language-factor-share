# LFS study guide: what to know, in what order, and how to defend it

**Written:** 2026-09-04
**For:** Waiz Khan, ahead of the ICLR 2027 submission (abstract 18 Sep, paper
25 Sep) and the meetings with Professor Koehn, Professor Murray, and other
JHU faculty and students.
**Scope:** LFS only. RMFS is out of scope by decision on 2026-09-04.
**Companions already in the repo:** `reports/lfs_math.pdf` (REML derivation
with a worked example), `presentation/LFS_Presentation_Script_and_Professor_QA_2026-08-28.docx`
(seven-slide script), `paper/LFS_EXPLAINED.tex` (plain-language two-pager),
`docs/AIM1_ASSESSMENT_FOR_ICLR_2026-09-04.md` (verdicts and numbers).

The guide has six parts. Part 1 is LFS itself, end to end, with a worked
example and every property proved in a few lines. Parts 2 to 4 are the
background you need to explain Part 1 to a machine-translation professor, an
information-retrieval professor, and a first-year PhD student. Part 5 is the
defense: the questions you will get and the answers that survive. Part 6 is
the study plan and reading list, ordered by payoff per hour.

---

## Part 1. How LFS works, end to end

### 1.1 The question in one sentence

When the same sentence is written in many languages and run through a model,
how much of the model's internal representation at a given layer is organized
by *which language* it is, and how much by *what it means*?

LFS answers that with one number between 0 and 1 per layer. Near 1: the
vectors cluster by language regardless of meaning. Near 0: the vectors
cluster by meaning regardless of language. The full set of numbers across
layers is the depth profile.

### 1.2 The data: a balanced grid

Take `N` sentences that exist in all `L` languages with the same meaning
(NTREX-128: the WMT19 news test set, professionally translated into 128
languages; we use `N = 300` sentences and `L = 128` languages for the
profiles). Every cell `(language l, sentence c)` is one string. That gives an
`L x N` grid of strings, and it is *balanced*: every language has every
sentence, every sentence has every language. Balance is what makes the
decomposition below exact.

### 1.3 From string to vector

For each string, run the model with hidden-state output enabled. A
decoder-only transformer keeps one vector of size `D` (the hidden size, for
example 1024 for Qwen3-0.6B, 4096 for Llama-3.1-8B) per token per layer.
Hugging Face returns `n_layers + 1` hidden states: index 0 is the output of
the embedding layer, index `k` is the residual stream after block `k`. The
project treats index 0 as "layer 0", which is why a 28-block model shows 29
layers in the results.

At one layer, average the token vectors over the sentence's real tokens (not
padding) to get one sentence vector `h[l, c]` in `R^D`. The averaging is done
in fp32 because some coordinates carry very large values ("massive
activations") whose sums overflow fp16.

Now the grid holds vectors: an `L x N x D` tensor.

### 1.4 Standardize each coordinate, jointly

Transformer coordinates have wildly different scales; a handful of
coordinates can carry values hundreds of times larger than the rest. If you
summed squared distances on the raw vectors, those coordinates would decide
the answer alone. So, for each coordinate `d`, subtract its mean and divide
by its standard deviation, where mean and standard deviation are taken over
**all `L x N` cells together**:

```
z[l, c, d] = (h[l, c, d] - mean_over_all_cells(h[., ., d])) / sd_over_all_cells(h[., ., d])
```

Two things to say when asked. First, this is a diagonal rescaling, and it
gives every coordinate the same total variance, so LFS becomes a variance
weighted average of per-coordinate language shares in which no coordinate can
dominate by scale alone. Second, standardization must be **joint**, never per
language: standardizing each language separately would subtract each
language's mean, which is exactly the quantity you are trying to measure.

### 1.5 Three kinds of averages

With `z` in hand, at one layer compute:

- the grand mean `mu`, the average of all `L x N` vectors;
- the language means `mu_l`, one per language, the average over its `N`
  sentences;
- the sentence (concept) means `mu_c`, one per sentence, the average over its
  `L` translations.

Averaging over sentences within a language cancels meaning and leaves
language identity. Averaging over languages within a sentence cancels
language and leaves meaning. That is the whole trick.

### 1.6 Sums of squares and the share

```
SS_lang = N * sum over l of ||mu_l - mu||^2       (how far language means sit from the center)
SS_con  = L * sum over c of ||mu_c - mu||^2       (how far sentence means sit from the center)
SS_res  = sum over l, c of ||z[l,c] - mu_l - mu_c + mu||^2   (what neither main effect explains)

LFS = SS_lang / (SS_lang + SS_con)
```

The multipliers `N` and `L` are bookkeeping, not choices: each language mean
stands for `N` cells and each sentence mean for `L` cells, so weighting them
that way makes the three terms add up exactly to the total spread of the grid
(Section 1.9).

The residual is left out of the ratio on purpose. LFS asks how the two
*systematic* effects divide. If the residual were in the denominator, a noisy
model would look language-agnostic simply by being noisy. The residual is
still computed and reported next to the ratio, because it tells you how much
of the grid the two-main-effect story does not cover.

### 1.7 A worked example you can do on a whiteboard

One coordinate (`D = 1`), two languages, three sentences:

```
                 s1    s2    s3    | language mean
English          1.0   3.0   5.0   |   3.0
German           2.0   4.0   6.0   |   4.0
-----------------------------------
sentence mean    1.5   3.5   5.5   |   grand mean 3.5
```

- `SS_lang = N * [(3.0 - 3.5)^2 + (4.0 - 3.5)^2] = 3 * (0.25 + 0.25) = 1.5`
- `SS_con  = L * [(1.5 - 3.5)^2 + (3.5 - 3.5)^2 + (5.5 - 3.5)^2] = 2 * (4 + 0 + 4) = 16`
- every residual `z - mu_l - mu_c + mu` is zero here (check one: `1.0 - 3.0 - 1.5 + 3.5 = 0`)
- `LFS = 1.5 / (1.5 + 16) = 0.086`

Meaning dominates: the sentences differ a lot (1, 3, 5) and the languages
differ by a constant offset of 1. Now multiply every number by 10. Every sum
of squares is multiplied by 100 and the ratio is unchanged. That is the scale
invariance that later becomes the collapse blind spot.

(The REML note in `reports/lfs_math.pdf` gets 0.11 on this same table because
it subtracts a noise estimate and divides by degrees of freedom. Same data,
different estimator. The paper uses the direct sums-of-squares form above;
REML is a robustness check. On 14 real models the two rank models identically.)

### 1.8 What the profile looks like and what it means

Compute LFS at every layer and plot it against relative depth. In all 17
primary models and all 19 expansion models the curve starts high, falls to an
interior minimum, and rises again toward the output. The reading:

- **Layer 0 is high** because the embedding layer knows the token identity,
  and tokens are language specific (different scripts, different subwords).
- **The interior minimum** is where the model represents the same meaning most
  similarly across languages. Its depth, `LFS(layer 0) - min LFS`, is the
  "dip depth". It runs from 0.039 (BLOOM-1.7B) to 0.336 (Qwen3-8B).
- **The rise toward the output** is because the model has to predict the next
  token, and the next token is in a specific language.

Two facts to state before anyone else does. First, the minimum value is still
above 0.5 for every model on this grid: even at the most shared layer,
language main effects exceed meaning main effects. Say "more meaning-weighted
relative to its own endpoints", never "concept-dominated". Second, the
location of the minimum varies: most models place it between 35 and 60
percent depth, but Mistral-7B (and two continued-pretraining models in the
expansion) place a sharp minimum at layer 2 followed by a plateau.

### 1.9 Properties, each with its short proof

**Bounded in [0, 1].** Both sums of squares are non-negative.

**Exact decomposition.** Write `a_l = mu_l - mu`, `b_c = mu_c - mu`,
`r[l,c] = z[l,c] - mu_l - mu_c + mu`, so `z[l,c] - mu = a_l + b_c + r[l,c]`.
On a balanced grid the `a_l` sum to zero, the `b_c` sum to zero, and the
residuals sum to zero along every row and every column. Expand the squared
norm of `a_l + b_c + r[l,c]` and sum over the grid: every cross term contains
one of those zero sums, so it vanishes. Hence
`total SS = SS_lang + SS_con + SS_res` exactly. This is the vector version of
balanced two-way analysis of variance with one observation per cell.

**Invariance to uniform scaling.** Multiplying every vector by `k` multiplies
every sum of squares by `k^2`; the ratio is unchanged. After joint
standardization LFS is also invariant to any positive rescaling of individual
coordinates, since standardization undoes it. Consequence: LFS cannot see a
model whose representations all shrink together.

**Invariance to rotation of the hidden basis.** A rotation preserves all the
norms in the sums, so the global LFS is unchanged. (Which coordinates carry
the language energy is not invariant; that is why "language neuron" claims do
not follow from this decomposition.)

**The finite-grid null.** Suppose there is no language structure and no
meaning structure, just independent noise with variance `sigma^2` in each of
`D` coordinates. The language means are averages of `N` noisy cells, so
`SS_lang` has expectation `(L - 1) * D * sigma^2`; the sentence means are
averages of `L` cells, so `SS_con` has expectation `(N - 1) * D * sigma^2`.
The ratio of expectations is

```
LFS_null = (L - 1) / (L + N - 2)
```

`D` cancels. At `L = 128, N = 300` this is `127 / 426 = 0.298`. At `N = 1500`
it is 0.078. Three consequences you must be able to recite: (i) absolute LFS
values are not comparable across grids of different `L` and `N`, or across
pooling schemes; (ii) differences at a fixed grid are fine, because the noise
term is common to both conditions; (iii) the null is not a floor, since strong
meaning structure with weak language structure pushes LFS below it. The
formula was verified on synthetic noise and on real representations by
splitting one language's sentences into pseudo-languages.

**Near-independence of evaluation size.** Because the noise term shrinks as a
share of the numerator when `N` grows, LFS drifts slightly downward with more
sentences: `-0.006` to `-0.010` from `N = 100` to `N = 1000`. A retrieval
score on the same draws moves `-0.065` to `-0.125`, because retrieval gets
harder with more candidates. This is the practical argument for LFS over
retrieval-based scores when comparing across studies.

**Differentiable.** LFS is a smooth function of the vectors, so it can be
used as a loss. That property is what made the Aim 2 test possible, and the
test showed it should not be used that way.

### 1.10 The construct boundary, in one paragraph

LFS measures the *additive* language component: how far each language's mean
sits from the center. It does not measure language-specific rotations,
warps, or language-by-meaning interactions; those land in the residual. We
measured how much that matters with a nested ladder of maps (offset, scale,
rotation, general linear, nonlinear) fit from each language into a consensus
space and scored on held-out sentences: offset plus scale account for 65 to
98 percent of cross-language misalignment, rotation under one percent, and
the instrument recovers injected rotations exactly, so the small rotation
share is evidence of absence. Language identity nonetheless remains almost
perfectly decodable by a linear probe at every layer (accuracy 0.875 to 0.886
against chance 0.042): a direction carrying one percent of the variance can
still be perfectly separable. Variance share and separability are different
quantities; LFS measures the first.

### 1.11 The failure mode you must own before anyone raises it

A ratio cannot see both terms shrinking together. In the Aim 2 word-alignment
arm, the representation norm fell from 451 to 42 and the raw concept variance
from 13,791 to 699 (a factor of twenty), while LFS moved from 0.445 to 0.493
and the retrieval metric MEXA ranked that collapsed model *first*. Every
scale-invariant reading shares this blind spot, by theorem. The fix is not a
better ratio; it is to report the ratio with its raw components
(`SS_lang`, `SS_con`, residual, representation norm) so that a collapse shows
up where it lives. Say this proactively. It is the strongest evidence in the
project that you understand your own instrument.

### 1.12 How LFS relates to behavior, and where it does not

- **Content transfer (positive).** Give a model an English sentence to
  predict, preceded by the previous sentence of the same document in another
  language, versus a wrong-document sentence in that language. The
  improvement from the right context is the content benefit. Across 19
  models and 13 families, models with a more meaning-weighted dip layer show
  more content transfer after controlling for data exposure, language
  family, script, and tokenizer fertility: Spearman 0.51, model-bootstrap
  95 percent interval [0.19, 0.71].
- **Exam accuracy (null).** Against pooled multiple-choice accuracy on 33
  models, LFS has no association once exposure is removed (0.04, interval
  crosses zero). Neither does anything else strongly: MEXA reaches 0.24.
  Two independent exams agree on only about a third of their per-language
  residual variation, which caps what any intrinsic metric can predict there.
- **Training objective (negative).** Minimizing LFS directly deepens the dip
  quickly but through a shortcut that damages alignment and capability; across
  18 trained arm-seed points, lower LFS went with *worse* tail alignment. The
  observational association reversed under intervention. LFS is a
  measurement, not a target.

---

## Part 2. How an LLM works, as much as you need

You do not need to derive backpropagation. You need to be able to say, with
confidence, where the numbers in the grid come from and why they behave the
way they do.

**Tokenization.** Text is split into subword tokens by a learned vocabulary
(BPE or SentencePiece). A language with little training data is split into
more, shorter pieces. *Fertility* is tokens per word (or per character)
relative to English; high fertility means the language pays more tokens for
the same meaning and gets less context per token. Fertility shows up in LFS
because mean pooling averages a different number of tokens per language and
because the tokenizer itself carries language identity. This is why fertility
is a control variable in every validity analysis.

**Embeddings and the residual stream.** Each token id indexes a row of the
embedding matrix, giving a `D`-dimensional vector. That vector is the start of
the *residual stream*: a running sum that every layer reads from and adds to.
Layer `k`'s hidden state is the residual stream after block `k`. This "add to
a shared vector" picture is the reason it is meaningful to compare states
across layers at all.

**A transformer block.** Attention plus a feed-forward network (MLP), each
wrapped with a normalization (RMSNorm in modern models) and a residual
connection. Attention lets each position pull information from earlier
positions (the query, key, value picture); the MLP transforms each position
on its own. Position is injected through rotary embeddings in most current
models. You should be able to draw this on a whiteboard in one minute.

**Output.** After the last block, a final norm and the unembedding matrix map
the `D`-vector to a score per vocabulary item (logits); softmax turns scores
into a probability distribution over the next token. Training minimizes the
negative log-likelihood (cross-entropy) of the actual next token under
*teacher forcing*, which means the model is always shown the true prefix.
Perplexity is `exp(mean NLL)`. The content-transfer harness uses exactly this
quantity: the NLL of the English target tokens given a context.

**Logit lens.** Apply the unembedding to an *intermediate* hidden state and
read off the most likely token. It is a heuristic, not a property of the
model, and it presumes that intermediate states live near the output
embedding space. We measured that presumption: mid-stack states reach a
maximum cosine of about 0.10 to any output embedding (0.20 at the last
layer). The proposal's skepticism about the lens was right, and now
quantified.

**Why layers differ.** Early layers resolve token identity and local
context, so their states are strongly language and script specific. Middle
layers hold the most abstract, most shared information. Late layers prepare
the next-token prediction, which is language specific again. The LFS profile
is a measurement of this story, and it also finds that the story has two
events: input script is dropped early (relative depth 0.05 to 0.19, seen with
the lens) while meaning converges later (0.3 to 0.5, seen with LFS).

**Precision.** fp16 has a small dynamic range; bf16 has fp32's range with
fewer mantissa bits; fp32 is the safe choice for accumulating sums. Massive
activations overflow fp16 sums, which is why pooling is always fp32 and why
BLOOM, trained in bf16, must never be run in fp16. Two project incidents came
from this; be ready to tell them.

**Base versus instruct.** All measured models are base checkpoints (next-token
predictors). Instruction tuning changes late layers most. Whether the dip
survives instruction tuning is an open question and a cheap experiment.

**Anisotropy and massive activations.** Transformer representations are not
spread evenly; a few coordinates carry enormous variance ("rogue
dimensions"), and cosine similarities between unrelated sentences are high.
This is why joint standardization is necessary, why whitening changes the
picture (language identity concentrates in the high-variance directions), and
why we measure in the model's native coordinate basis and say so.

---

## Part 3. Machine translation and multilingual NLP, as much as you need

Professor Koehn's field. Speak it correctly.

**Parallel corpora and n-way parallel test sets.** A parallel corpus pairs
sentences with their translations. NTREX-128 and FLORES-200 are n-way
parallel: the same source sentences in over a hundred languages, so any pair
of languages is aligned through the source. This is what makes the balanced
grid possible. Know the numbers: NTREX has 1,997 sentences in 123 documents,
128 languages; we use the first 300 sentences for profiles and 1,500 for the
saved-representation campaign.

**Translationese.** Translated text carries the shadow of its source
language: more literal word order, source-like lexical choices, simplified
style. Because NTREX is translated *from* English, every non-English cell is
translationese. The proposal itself criticizes benchmarks built this way. We
tested the effect on WMT19 pairs measured in both directions: translated text
shows a slightly higher language share (+0.02 to +0.06) and slightly lower
retrieval alignment, but model ordering is preserved at Spearman 0.99, so
comparative conclusions survive. Say the number, not the hand-wave.

**Direction matters.** "Native side" versus "translated side" of a test set
is a real distinction in MT evaluation. WMT test sets since 2019 include
both. This is why the translationese check could be run at all.

**Resource tiers, families, scripts.** High-, mid-, and low-resource is
about training data availability, usually proxied by web-corpus size
(CulturaX, CC-100). Language family (Indo-European, Sino-Tibetan, Niger-Congo)
and script (Latin, Cyrillic, Devanagari, Arabic, Han) are the standard
confounds in cross-lingual work because they predict both alignment and task
performance. Our deflation regression controls for all four: log tokens,
macro-family, script, fertility.

**Cross-lingual word embedding alignment.** The classic result (Mikolov et
al. 2013; Conneau et al. 2018, MUSE; Artetxe et al.) is that word-embedding
spaces of two languages can be aligned with a single linear, often
orthogonal, map (Procrustes). Koehn's group did extensive work here
(Marchisio et al. 2020 to 2022). The proposal asks whether that still holds
inside deep LLM layers; our misalignment budget answers: mostly an additive
offset plus scale, rotation negligible, nonlinear residual small.

**Hubness.** In high dimensions some points are nearest neighbors of many
others. In cross-lingual retrieval, frequent English words become hubs. The
proposal suspects hubness inflates the "LLMs think in English" reading. Our
hub test asks, for each language, whether English or a leave-one-out
multilingual average predicts its representations better: the multilingual
factor wins for 113 to 124 of 127 languages in every model.

**Retrieval-based alignment metrics.** MEXA (Kargaran et al. 2025) checks
whether each sentence's translation is its nearest neighbor among
candidates (strict row and column dominance). Alignment-at-Risk (AaR@10,
project metric) averages the retrieval margin of the worst decile of
sentences. Both depend on the number of candidates and both are blind to
uniform collapse. Know how each is computed so the comparison is fair.

**Downstream benchmarks.** Belebele: 4-choice reading comprehension in 122
languages, translated from English, chance 0.25. INCLUDE: natively authored
exams in 44 languages. Know the protocol issue we found: five-shot prompts
silently truncate on 2k to 4k context models, so everything is zero-shot.

**Word alignment.** eflomal (statistical) and SimAlign (embedding-based) link
words across a sentence pair. Two aligners agreeing is a reliability check.
Agreement tracks typology: agglutinative languages align poorly because one
word carries what English spreads across several. Relevant if anyone asks
about the word-level replication, which is boundary evidence only.

**Code-switching, parallel data in pretraining, alignment losses.** Know the
Aim 2 landscape from the proposal: parallel data drives cross-lingual
ability (Briakou et al. 2023); code-switched data, sentence-level alignment
losses (Liu and Niehues 2025; Bu et al. 2025), word-level losses, adversarial
language discriminators. You tested three of the four naive forms; none
helped and one degenerated.

---

## Part 4. Statistics and measurement, as much as you need

**Variance decomposition (ANOVA).** Total spread of a set of numbers splits
into between-group and within-group parts; with two crossed factors on a
balanced design it splits into factor A, factor B, and residual, exactly,
because the effects are orthogonal. LFS is this with vectors. Degrees of
freedom: `L - 1`, `N - 1`, `(L - 1)(N - 1)`. Expected mean squares give
noise-corrected variance components; the REML estimator uses them. Be ready
for "this is just ANOVA": yes, and the contribution is the operationalization,
the null, the validation, and the boundaries.

**Standardization and invariances.** Z-scoring is an affine map per
coordinate. Know what LFS is invariant to (uniform scale, coordinate
rescaling after standardization, rotation) and what it is sensitive to (the
choice of basis before standardization, pooling, grid size).

**Correlation.** Pearson measures linear association; Spearman is Pearson on
ranks and is robust to monotone transforms and outliers. We report Spearman
throughout. Know that a correlation of 0.5 explains about a quarter of the
variance.

**Confounding and deflation.** If languages with more data have both higher
alignment and higher accuracy, a raw correlation between the two is partly
"data size predicting data size". Residualize both variables on the
observables (log tokens, family, script, fertility) with model fixed effects
and correlate the residuals. That is the deflation. Raw 0.50 to 0.54 became
0.19 to 0.31: roughly half the field's usual number is confound. Know the
regression form: outcome on the logit scale above the chance floor, model
fixed effects, cluster-robust errors by language, and the sanity gate that
the token coefficient must be positive and significant before any residual is
interpreted.

**Fixed effects.** A dummy variable per model absorbs each model's overall
level, so comparisons are within model across languages.

**Bootstrap and clustering.** Resample to get confidence intervals. The unit
you resample must be the unit you want to generalize over. Languages within a
model are correlated, so resampling rows overstates certainty; resample
models to generalize to new models, languages to generalize to new languages,
or both (crossed). Four models cannot support a model-level claim; nineteen
can. This is the single most important statistical point in the paper.

**Effective sample size.** Cross-language correlation means 40 languages
carry the information of far fewer independent ones (about 1.2 to 2 in our
panels). Nominal `n` overstates evidence.

**Reliability and ceilings.** If two independent exams of the same thing
agree at 0.315 on their residuals, the best any predictor can do against one
of them is about `sqrt(0.315) = 0.56`. Spearman-Brown: averaging parallel
forms raises reliability. This is how the exam null was interpreted.

**Multiple comparisons.** Benjamini-Hochberg controls the false discovery
rate over a pre-registered set of tests.

**Preregistration.** Write the prediction, the criterion, and the analysis
before seeing the outcome, in a dated file. Report every failure. This is
why the paper can say "predicted before observed" about the collapse blind
spot, and why the discrepancy ledger exists. It is also the answer to "did
you tune this?"

**Construct validity, predictive validity, Goodhart.** Construct validity:
does the number measure what it says (the decomposition, the null, the
invariances, the probe contrast). Predictive validity: does it relate to
behavior (content transfer yes, exams no). Goodhart's law: when a measure
becomes a target it stops being a good measure; the Aim 2 reversal is a
textbook instance, and BLEU is the familiar analogue in MT.

**Linear algebra you will be asked about.** Norms and cosine similarity;
projection and least squares (the hub test's ridge regression, the mapping
ladder); orthogonal matrices and Procrustes; PCA and SVD; effective rank and
participation ratio (how many directions carry the variance); whitening
(ZCA) as "make the covariance identity"; Gini coefficient as a concentration
measure.

**Interpretability method basics.** Linear probes (train a classifier on
hidden states; high accuracy means the information is linearly present);
activation patching (swap a state from one run into another and see what
changes; Dumas et al. 2025 use it to separate language from concept
causally); representational similarity (CKA, invariant to orthogonal maps
and scale). Know what each can and cannot conclude.

---

## Part 5. The defense: questions and answers

Below are the questions a machine-translation professor, an IR professor,
and a sharp student will ask, in rough order of likelihood. Each answer is
one you can give without notes. Numbers are the audited ones.

**"Isn't this just two-way ANOVA on embeddings?"**
Yes, the decomposition is standard. The contribution is what you do with it:
a balanced translation grid that makes it exact, a closed-form null that
makes values interpretable, 33 models that show the profile is universal in
shape and nine-fold variable in depth, deflated behavioral validation, and
stress tests that show exactly when the ratio misleads. Nobody had done the
last three for any intrinsic multilinguality metric.

**"The U-shape is known."**
Qualitatively, yes, and I cite that work. What is new is the number: a
comparable depth that varies from 0.04 to 0.34 across models and tracks
training recipe more than parameter count; the finding that the dip is not a
scaling proxy (Salamandra shallows with scale while alignment improves); the
two-event separation of script and meaning; and the fact that three models
put the minimum at layer 2, which no "middle layers are shared" summary
predicts.

**"Does it predict downstream performance better than MEXA?"**
No, and I say so. On pooled exam accuracy neither does well after removing
data exposure, and MEXA is somewhat better. On content transfer MEXA's point
estimate is higher too. LFS is not a better predictor; it is a different
instrument: it decomposes, it is stable to evaluation size where retrieval
scores move ten times more, it has a known null, and it reads a structural
property rather than a nearest-neighbor outcome. The paper is about measuring
structure, not about winning a leaderboard.

**"Why is LFS above 0.5 even at the dip? Where is the shared space?"**
With 128 languages the language offsets (script, tokenizer, high-variance
directions) are large relative to the spread of 300 sentence means, and the
numerator carries a noise term that never vanishes. The correct reading is
relative: the dip layer is more meaning-weighted than the model's own
endpoints, and the null at this grid is 0.298, so 0.6 is far from noise. I do
not say "concept-dominated".

**"Why standardize? Why jointly?"**
Because a few coordinates carry values hundreds of times larger than the
rest and would decide the answer alone. Jointly, because per-language
standardization would subtract the very language means we are measuring.
I also report the whitened variant, which reshuffles the middle of the
ranking, so the basis choice is disclosed as a choice.

**"Why leave the residual out of the denominator?"**
LFS asks how the two systematic main effects divide. With the residual in the
denominator, a noisy model would look language-agnostic by being noisy. The
residual is reported beside the ratio, and it is where rotations and
interactions would show up.

**"Mean pooling injects sentence length and tokenizer effects."**
Measured. Absolute dip depth moves by 0.13 to 0.20 across mean, final-token,
unit-norm, and length-matched pooling; cross-model ordering changes by at
most one adjacent swap; and the within-family result holds for two Qwen3
sizes that share a tokenizer. Fertility correlates with each language's
contribution to the language variance at +0.42 to +0.64, so part of the
language component is tokenization, and I say so.

**"Your data is translated from English on both sides."**
Tested directly on WMT19 pairs in both directions: translated text shifts
the level slightly (+0.02 to +0.06 language share) and leaves model ordering
at Spearman 0.99. On the exam side, a natively authored exam gave the same
null as the translated exam restricted to the same languages, so the null is
about the narrow resource range of exam languages, not translation.

**"Can I train with it?"**
Not by itself. Minimizing LFS finds a shortcut: it deepens the dip while
damaging alignment and capability, and the word-level alignment loss in the
proposal collapses representations by a factor of twenty while LFS barely
moves and MEXA ranks the collapsed model first. Any training use needs the
raw components as constraints and independent capability checks. This is the
Goodhart result and it is central, not incidental.

**"Is the dip causal? Does a deeper dip make a model better?"**
No causal claim. Across pretrained models a deeper dip goes with more
cross-lingual content use, after controls, on 19 models. Under intervention
the association reversed. Observation and intervention are different
questions and I report both.

**"Only Qwen models drive this."**
Leave-one-family-out estimates stay between 0.44 and 0.60 for the content
result; the family bootstrap excludes zero; 13 families are in the panel.

**"What about instruction-tuned models, other corpora, larger models?"**
Open. All measured models are base checkpoints up to 13B; FLORES-200
replication and instruct variants are forward-pass-only experiments I can run
if useful. The 33-model result is on one corpus.

**"Why not a single number per model?"**
Because the readings disagree in exactly the cases that matter: a collapsed
model has a fine ratio and a destroyed raw component; a healthy model can have
a shallow dip. The deliverable is a profile: LFS plus `SS_lang`, `SS_con`,
residual, norm, and a behavioral check chosen for the question.

**"What would change your mind?"**
A model with a deep dip and poor content transfer after controls, at scale;
a corpus on which the profile shape does not hold; or a pooling scheme under
which the cross-model ordering breaks. I looked for all three and report what
I found.

**"What is the one-sentence claim?"**
LFS is a reproducible layerwise decomposition of language and concept
structure in multilingual representations whose profile generalizes across 33
models and carries target-specific behavioral signal, and whose measured
limits show why it must be reported with its raw components rather than used
as a score or a loss.

---

## Part 6. Study plan and reading list

Order is by payoff per hour for the next three weeks. Each block names what
to read and what you should be able to do afterwards.

### Block A (days 1 to 3): own the instrument

Do: rederive Section 1.7 by hand; rederive the null (Section 1.9); read
`code/pilot_metrics.py:lfs` line by line and map each line to the formulas;
run the worked example in numpy; explain the profile of one model from
`results/grid/<model>/metrics.json` layer by layer.
Read: `reports/lfs_math.pdf`; the Aug 28 presentation script; the paper draft
Sections 3 and 5.

### Block B (days 3 to 6): the transformer, precisely

Read: Jurafsky and Martin, *Speech and Language Processing* (3rd ed. draft),
the chapters on transformers and large language models; Vaswani et al.
(2017) "Attention Is All You Need"; Elhage et al. (2021) "A Mathematical
Framework for Transformer Circuits" (the residual-stream picture only);
nostalgebraist (2020) on the logit lens; Timkey and van Schijndel (2021)
"All Bark and No Bite" on rogue dimensions; Sun et al. (2024) "Massive
Activations in Large Language Models".
Be able to: draw a block, explain hidden states per layer, explain why fp32
pooling, explain what the lens assumes and why the 0.10 cosine matters.

### Block C (days 6 to 10): statistics of measurement

Read: any introductory ANOVA chapter (two-way, balanced, expected mean
squares); Searle, Casella, and McCulloch, *Variance Components* (chapter on
balanced designs, for REML); Efron and Tibshirani, *An Introduction to the
Bootstrap* (chapters 6 and 7; then look up cluster bootstrap); Gelman and
Hill, *Data Analysis Using Regression and Multilevel/Hierarchical Models*
(fixed effects, clustered data); a short note on Benjamini-Hochberg; Nosek et
al. (2018) "The preregistration revolution"; Strathern (1997) on Goodhart's
law.
Be able to: explain why four models cannot support a model claim, what
deflation removes, what the reliability ceiling means, and what
preregistration bought this project.

### Block D (days 8 to 14): the MT and multilingual literature Koehn lives in

Read: Koehn, *Neural Machine Translation* (2020), chapters on the transformer,
multilingual models, and evaluation (know his framing); Federmann, Kocmi, and
Xin (2022) NTREX-128; the WMT19 findings paper (for translationese and test
set directions); Conneau et al. (2018) MUSE and Artetxe et al. (2018) on
cross-lingual embedding alignment; Marchisio et al. (2020 to 2022) from
Koehn's group; Kargaran et al. (2025) MEXA in full; Bandarkar et al. (2024)
Belebele; Romanou et al. (2025) INCLUDE; Ahia et al. (2023) and Petrov et al.
(2023) on tokenizer fertility and fairness; Briakou, Cherry, and Foster
(2023) on incidental bilingualism.
Be able to: talk about parallel data, translationese, direction, fertility,
and embedding-space mapping in Koehn's vocabulary.

### Block E (days 10 to 16): the interpretability debate you are entering

Read: Wendler et al. (2024) "Do Llamas Work in English?"; Shani and Basirat
(2025) on language dominance; Tezuka and Inoue (2025) transfer neurons; Zhao
et al. (2024) "How do LLMs handle multilingualism?"; Bafna et al. (2025) from
Murray's group; Dumas et al. (2025) "Separating Tongue from Thought";
Kornblith et al. (2019) CKA; Chang, Tu, and Bergen (2022) geometry of
multilingual representations; Zhao et al. (2025) Lens (cited in the proposal).
Be able to: state each paper's claim in one sentence and say exactly what LFS
adds or contradicts.

### Block F (ongoing): ML foundations, for the long run

Read: Goodfellow, Bengio, and Courville, *Deep Learning*, chapters 5 to 8
(ML basics, MLPs, regularization, optimization) and 10 (sequence models);
Murphy, *Probabilistic Machine Learning: An Introduction*, chapters on linear
models and inference; a linear algebra refresher (Strang or the *Deep
Learning* book's chapter 2) on projections, orthogonal matrices, SVD, and
PCA.
Be able to: explain gradient descent, overfitting and held-out evaluation,
what a loss function is, why differentiability matters, and what SVD does.

### The concept checklist

Tick each when you can explain it to a first-year student in two minutes
without notes.

*LFS itself:* balanced grid; mean pooling in fp32; joint z-scoring; grand,
language, and sentence means; `SS_lang`, `SS_con`, `SS_res` and the
multipliers `N`, `L`; the ratio; the residual's role; the exact decomposition
proof; boundedness; scale invariance; rotation invariance; the null
`(L-1)/(L+N-2)` and its three consequences; near size-independence; dip
depth and endpoint recovery; why values stay above 0.5; the construct
boundary (additive); variance share versus separability; the collapse blind
spot and the profile fix; observational versus interventional; REML variant
as robustness.

*LLMs:* tokenization and fertility; embeddings; residual stream; attention,
MLP, norm; hidden states per layer; unembedding, logits, softmax; NLL,
teacher forcing, perplexity; logit lens and its validity limit; base versus
instruct; fp16, bf16, fp32; massive activations and anisotropy.

*MT and multilingual NLP:* parallel and n-way parallel corpora; NTREX,
FLORES, WMT; translationese and direction; resource tiers, families,
scripts; cross-lingual embedding alignment and Procrustes; hubness; MEXA and
retrieval margins; Belebele and INCLUDE; word alignment; the Aim 2 landscape
(parallel data, code-switching, alignment losses).

*Statistics:* two-way ANOVA and degrees of freedom; expected mean squares
and variance components; Spearman versus Pearson; residualization and
deflation; fixed effects; cluster bootstrap and the unit of generalization;
effective sample size; reliability ceilings and Spearman-Brown;
Benjamini-Hochberg; preregistration; construct and predictive validity;
Goodhart's law.

*Linear algebra and interpretability:* norms and cosine; projection and
ridge regression; orthogonal maps; PCA and SVD; effective rank and
participation ratio; whitening; Gini; linear probes; activation patching;
CKA.

### How to practice

Give the ten-second, thirty-second, and two-minute versions (Part 1, and the
Aug 28 script) out loud once a day. Once a week, have someone read Part 5
questions to you in random order. Before the Koehn meeting, redo the
whiteboard example cold. Every number you say should be one you can point to
in `docs/AIM1_ASSESSMENT_FOR_ICLR_2026-09-04.md` or the claims ledger.
