# Word-Level Stage 0: Alignment Reliability

**Run 2026-07-24. Twenty languages, eleven scripts, 300 sentences each,
English pivot. No LLM representations were computed.**

## Verdict

**The stage gate FAILS at its stated threshold.** The gate required
inter-aligner agreement above 0.70 F1 on high-resource pairs. Of seven
high-resource languages, **two pass** (Spanish 0.778, German 0.745); mean
agreement is 0.636 with a range of 0.502 to 0.778. The threshold has not
been moved.

Across the full set, four of twenty languages clear 0.70: Indonesian,
Spanish, German, Czech. All four are Latin-script and analytic.

## Method

Two aligners sharing no machinery, so their agreement is a reliability
estimate rather than two views of one model:

- **SimAlign** over XLM-RoBERTa embeddings, no fine-tuning, intersection
  method (high precision). Substituted for awesome-align, which is a
  research repository pinned to an old transformers version; same underlying
  method and published. Recorded rather than silent.
- **eflomal**, statistical, IBM-model style, fitted unsupervised on the
  parallel text itself (2000 sentence pairs), symmetrized by intersecting
  forward and reverse directions.

Content words are defined on the English pivot side only, so one closed-class
list suffices and no per-language lexical resource is needed. Function words
are counted separately rather than discarded.

## Result 1: agreement tracks typology, not noise

| Group | Languages | F1 | Interpretation |
|---|---|---|---|
| Latin, analytic | ind 0.817, spa 0.778, deu 0.745, ces 0.701 | high | alignment works |
| Non-Latin, alphabetic | khm 0.696, hin 0.643, zho 0.604, ben 0.587, rus 0.583, kat 0.562, arb 0.553 | moderate | harder, but the unit is real |
| Morphologically rich | tur 0.497, amh 0.403, zul 0.326 | low | see below |
| No segmenter available | mya 0.131 | broken | tokenization, not alignment |

**The morphology result is the scientifically important one and is not
fixable by better tooling.** Turkish, Amharic, and Zulu are the three lowest
non-tokenization cases. In an agglutinative language a single word carries
what English distributes across several, so a one-to-one word correspondence
frequently does not exist to be found. This is a property of the word as a
cross-lingual unit, not a deficiency of either aligner. Note that Zulu is
Latin-script and Turkish is Latin-script, so this is not a script effect.

Function-word coverage is below content-word coverage in every one of the
twenty languages, which is why they were kept in a separate column rather
than discarded.

## Result 2: unsegmented scripts failed for a diagnosable reason, and it was fixed

Whitespace tokenization returns whole clauses as single tokens for scripts
that do not mark word boundaries. The token counts made the mechanism
unambiguous before any aligner was blamed: Japanese averaged 1.2 tokens per
sentence against English's 20.

| Language | Segmenter | Tokens/sentence | Agreement F1 | Content coverage |
|---|---|---|---|---|
| Chinese | jieba | 2.7 to **19.2** | 0.118 to **0.604** | 0.059 to **0.638** |
| Japanese | fugashi + unidic-lite | 1.2 to **26.5** | 0.096 to **0.502** | 0.005 to **0.542** |
| Khmer | khmer-nltk | 5.2 to **25.3** | 0.199 to **0.696** | 0.155 to **0.679** |
| Burmese | none available | 11.6 (unchanged) | 0.131 (unchanged) | 0.337 (unchanged) |

A five-fold improvement for Chinese and Japanese and a three-and-a-half-fold
improvement for Khmer, with Burmese unchanged as the control that no
segmenter was applied to. This is the evidence that the original failure was
tokenization rather than alignment difficulty. The whitespace results are
retained in `coverage_whitespace_baseline.json`.

**Caveat carried forward.** For these languages the word is a segmenter's
decision, not an orthographic fact, and different segmenters disagree. A
Chinese word-level unit is therefore not the same kind of object as a German
one. The segmenter used is recorded per language in the output.

## Consequence for word-level work

The honest reading is not that the aligners are inadequate. It is that
**the word is a reliable cross-lingual unit only within a narrow typological
band**, and outside that band its reliability degrades in a way that is
predictable from morphology and orthography rather than from resource level.
Two of the four languages clearing the gate are mid-resource, and Swahili, a
low-resource language, outperforms Russian and Arabic.

Any word-level decomposition that proceeds must therefore declare its
language subset in advance and report per-language alignment coverage
alongside every number, because coverage varies from 0.83 to 0.30 across the
set and a measurement computed on 30 percent of tokens is not comparable to
one computed on 83 percent.

This also bears on Aim 2 of the proposal, which contemplates word-level
alignment objectives. The measurement here says such an objective would apply
reliable supervision to analytic languages and increasingly noisy supervision
to agglutinative ones, which is the opposite of what a multilingual training
objective wants.

## Artifacts

- `results/wordlevel/coverage_final.json` (merged, best tokenization)
- `results/wordlevel/coverage_whitespace_baseline.json` (before segmenters)
- `results/wordlevel/segmentation_effect.json` (the before/after table)
- `results/wordlevel/alignments/<lang>.json` (per-language links)
- `src/wordlevel/align.py`, `configs/wordlevel/prototype20.yaml`
