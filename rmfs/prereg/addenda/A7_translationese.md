# Addendum A7 — does translationese drive the readings?

**Dated 2026-08-07, written BEFORE the run. The proposal commits to
evaluation that avoids translationese; every number in this project so
far rests on translated text (NTREX is English-source translated into
128 languages; Belebele is translated from English passages). This
addendum tests directly whether that dependence changes what the
readings say.**

## Design

WMT19 test sets are partitioned by the original language of each
document, so for the same language, same domain and same year we can
obtain both directions:

- **native L**: from the `L-en` set, the L side was written by a native
  author and the English side is a translation of it;
- **translated L**: from the `en-L` set, the English side is original
  and the L side is a translation (translationese).

Languages available in both directions: de, ru, zh, fi, lt, gu, kk (cs has only one
direction in WMT19 and is excluded).
N = 300 sentence pairs per language per direction, all 14 grid models,
each model's own frozen layer, identical preprocessing to the main
pipeline.

Three readings are computable from a bilingual pair and are compared
between the two directions:

1. **Sentence matching** (retrieval of the parallel sentence);
2. **Meaning against language** (variance decomposition over the two
   languages);
3. **Weakest aligned tail**.

The collapse flag needs a reference checkpoint and is out of scope here.

## Frozen predictions

- **T1 (direction effect on matching).** Sentence matching is HIGHER
  when the target language side is translationese than when it is
  native, in at least 5 of the 7 languages. Translations track their
  source more literally, so matching should be easier.
- **T2 (direction effect on the split).** The language share is LOWER
  on the translationese side, in at least 5 of 7 languages, for the
  same reason.
- **T3 (ordering, the decision relevant one).** Model ordering is
  preserved across the two kinds of text: Spearman correlation between
  the per-model reading computed on native text and the same reading
  computed on translated text is **>= 0.8** for each of the three
  readings.
- **T4 (relative magnitude).** For each reading, the mean absolute
  native-to-translated gap is SMALLER than the spread across the 14
  models on native text alone, i.e. translationese shifts the level but
  does not dominate the comparison between models.

T1 and T2 characterise the artefact. **T3 and T4 decide whether the
project's conclusions survive it.** If T3 or T4 fails, the finding is
that metric comparisons in this literature, ours included, are
confounded by translation direction, and it is reported as such.
