# Addendum A8 — do the readings predict performance on natively
# authored tasks?

**Dated 2026-08-07, written BEFORE the run. A7 showed that the
direction of translation shifts the readings only slightly and leaves
model ordering intact. A7 did NOT test the extrinsic side: every
validity number in this project is measured against Belebele, which is
translated from English source passages. This addendum tests the
readings against a benchmark whose questions were never translated.**

## Why A7 was not sufficient

A7 compared two parallel corpora that differ in which side was
authored first. It could not compare translated text against text with
no translation anywhere, because parallel data always involves
translation on one side — that is a structural property of the problem,
not an oversight. The extrinsic side has no such constraint: a
downstream benchmark can be natively authored end to end.

## Benchmark

**INCLUDE** (`CohereForAI/include-base-44`): multiple choice questions
drawn from regional academic and professional examinations, written in
the local language for local test takers. No translation step. About
500 questions per language.

30 languages overlap with the panel used throughout this project:
arabic, bengali, bulgarian, chinese, dutch, estonian, finnish, french,
georgian, german, greek, hebrew, hindi, hungarian, indonesian, italian,
japanese, korean, malay, persian, polish, portuguese, russian, spanish,
tamil, telugu, turkish, ukrainian, urdu, vietnamese.

Mapping to the panel codes is fixed here: arabic=arb, bengali=ben,
bulgarian=bul, chinese=zho-CN, dutch=nld, estonian=est, finnish=fin,
french=fra, georgian=kat, german=deu, greek=ell, hebrew=heb, hindi=hin,
hungarian=hun, indonesian=ind, italian=ita, japanese=jpn, korean=kor,
malay=msa, persian=fas, polish=pol, portuguese=por, russian=rus,
spanish=spa, tamil=tam, telugu=tel, turkish=tur, ukrainian=ukr,
urdu=urd, vietnamese=vie.

Scoring is zero shot by log likelihood, all 14 grid models, matching
the Belebele protocol.

## Frozen predictions

- **N1 (sanity).** Aggregate accuracy over the 30 languages is above
  chance for at least 10 of the 14 models. Models at or below chance
  are excluded from the validity comparison, by the same rule used for
  the translated exam.
- **N2 (the key one).** The sentence matching reading predicts native
  exam performance after the standard controls, with the 95 percent
  interval excluding zero. On the translated exam this reading gives
  0.243 [0.172, 0.289].
- **N3 (is the translated exam a faithful proxy).** Per model accuracy
  on the native exam correlates with per model accuracy on the
  translated exam at Spearman >= 0.8 across the 14 models.
- **N4 (no inflation).** The deflated validity measured on the native
  exam falls inside the 95 percent interval of the translated exam
  estimate, that is, translation of the benchmark did not inflate the
  reported validity.

N2 and N4 decide whether the validity claims in this project are an
artefact of translated evaluation. A failure of N4 in the downward
direction would mean the numbers reported so far are too high and must
be restated; that outcome is reported as scored, like every other.
