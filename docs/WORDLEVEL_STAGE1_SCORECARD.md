# Word-Level LFS, Stage 1: Scorecard

**Final run 2026-07-27 on the full NTREX corpus (1997 sentences), against
`prereg/PREREG_WORDLEVEL.md` and addendum W1.** Qwen3-1.7B, dip layer 13
and its two neighbours on each side, three subword-pooling variants, two
declared operating points. Supersedes two earlier runs, both superseded for
recorded reasons (D5 function-word contamination, W1 corpus size).

## Result

Both operating points are now interpretable and every gate passes. The
headline directional prediction fails at both, for a reason established by
a separate control rather than by linguistics.

| | 11 languages | 4 languages |
|---|---|---|
| content concepts | 1081 | 5117 |
| estimator pure-noise value | 0.0101 | 0.0008 |
| word-level LFS at the dip layer | 0.0686 (**7x** the floor) | 0.0564 (**72x** the floor) |

## Gates

| Gate | 11 languages | 4 languages | Verdict |
|---|---|---|---|
| **G-WL.1** reconstruction, Pearson > 0.80 | **+1.000** | **+0.998** | **PASS** |
| **G-WL.2** monolingual null at the pure-noise value, within 0.02 | 0.1086 vs 0.1100 | 0.0038 vs 0.0039 | **PASS** |
| **G-WL.3** permutation null, signal-present direction | null 0.3152, real 0.0686 | null 0.1522, real 0.0564 | **PASS** |

G-WL.2 is worth stating plainly: the monolingual null lands within 0.0014
and 0.0001 of the predicted degrees-of-freedom value, at two grids whose
predicted values differ by a factor of 28. The characterization derived at
sentence level transfers to word level exactly.

G-WL.3's inequality was written backwards in the pre-registration and is
recorded as D4; the data pass overwhelmingly in the correct direction, real
values sitting three to five times below the null.

## Predictions

| Prediction | 11 languages | 4 languages | Verdict |
|---|---|---|---|
| **P-WL.1** word-level exceeds sentence-level | 0.84x | 0.51x | **FAIL** |
| **P-WL.2** U-shape with interior minimum | min at L13 | min at L13 | **PASS** |
| **P-WL.4** the two points agree in direction | agree on both P-WL.1 and P-WL.2 | | **PASS** |
| **P-WL.5** pooling changes value, not sign or location | all three poolings minimize at L13 | all three at L13 | **PASS** |
| **P-WL.3** language ordering agreement | not computed | | not scored |

## What stands

**The U-shape replicates across granularity, at the same layer.** Word-level
LFS is minimized at layer 13, which is the layer the sentence-level analysis
independently identifies as the model's dip. This holds at both operating
points and under all three subword-pooling schemes, six configurations in
total. The most language-neutral depth of the network is the same whether
the unit of measurement is a whole sentence or a single aligned word. That
is a genuine cross-granularity replication and it is the substantive
positive result of this stage.

## What does not stand, and why

Word-level LFS is *lower* than the matched sentence-level value at both
points, 0.84x and 0.51x, the opposite of the prediction. The tempting
reading is that language identity lives in composition rather than in
individual content words.

**That reading is not supported.** A control on synthetic data with a fixed
per-language offset, independent concept noise, and no linguistic structure
whatsoever shows that averaging items inflates the measured language share
by itself: 3.0x at four items, 4.6x at eight, 6.3x at sixteen. The observed
real-data ratios are smaller than this structureless baseline produces at
comparable counts. The direction of the failure is explained, and arguably
over-explained, without appeal to language.

The durable consequence is methodological: **LFS values are not comparable
across pooling granularities.** This joins evaluation-set size as a second
dimension along which absolute values must not be placed side by side, and
it sharpens the queued fertility work, since languages differ systematically
in how many subword tokens their sentences carry.

## Corrections behind this run

Two earlier versions were superseded, both for reasons recorded in the
discrepancy ledger rather than discovered by the checks that were designed.

- **D5.** The concept grid contained 35 to 39 percent function words,
  contrary to the pre-registration. The content filter existed only in the
  coverage reporting path and was never applied to the link set. Removing
  them cut the 11-language grid by 39 percent and, at 300 sentences, pushed
  that operating point below its own noise floor.
- **W1.** Stage 0 aligned 300 sentences, inherited from the sentence-level
  convention. At word level the concept set is an intersection across
  languages and shrinks multiplicatively, so 300 sentences left too few
  concepts to support 11 languages. Re-running on all 1997 sentences raised
  the count from 155 to 1081 and dropped the floor from 0.067 to 0.010.

## Scope

One model. The 4-language point is the more precise (72x headroom, 5117
concepts); the 11-language point is the broader (7x headroom) and is the
declared headline. Stage 2 (word types, SenseShare) remains blocked by type
inventory: even at full corpus the number of content-word types with enough
repeated occurrences needs re-checking before it can be attempted.
