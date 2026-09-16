# Fertility and Pooling Robustness: Scorecard

**Run 2026-07-27 against `docs/FERTILITY_ROBUSTNESS_PLAN.md`, frozen
before any pooling-variant run.** Four pooling schemes on identical
sentences and languages, so every difference is attributable to pooling
alone. All five models complete.

## The frozen decision rule fails for every variant

| Variant | Cross-model Spearman of dip depth (gate 0.90) | Max abs. change in dip depth (gate 0.03) | Verdict |
|---|---|---|---|
| final-token | +0.900 FAIL (at the boundary) | 0.143 FAIL | **FAIL** |
| unit-norm then mean | +0.900 FAIL (at the boundary) | 0.132 FAIL | **FAIL** |
| length-matched | +0.600 FAIL | 0.199 FAIL | **FAIL** (declared a bound) |

Both non-bound variants land at 0.900 against a gate of "above 0.90", so
they fail by the narrowest possible margin. That is reported as a fail
because the gate was written as a strict inequality before the run. On the
four models available before Qwen3-8B completed, final-token pooling scored
a perfect +1.000; adding the fifth model moved it to the boundary.

**The magnitude of dip depth is strongly pooling-dependent.** Absolute
values move by up to 0.20, which is comparable to the entire range across
models. The reported range of 0.04 to 0.37 is therefore a property of
mean-pooled representations, not of the models alone, and must be stated
that way.

## What survives, and the reason it is informative

Magnitude and ranking come apart. Under **final-token pooling the cross-model
ranking is very nearly preserved** (Spearman +0.900, a single adjacent
transposition among five models).

That specific result carries weight beyond its number. Final-token pooling
takes a single position and has **no length denominator at all**, so it is
structurally immune to the mechanism the fertility concern proposes: that
averaging over more tokens in high-fertility languages suppresses
concept-idiosyncratic variance and inflates the language share. If the
cross-model ordering were an artifact of that mechanism, the scheme without
the mechanism should not reproduce the ordering. It reproduces it perfectly.

Ranking robustness degrades in the expected order: +0.900 under final-token
and unit-norm, +0.600 under length-matching, which alters the text being
encoded rather than only the denominator and was declared a bound in advance
for that reason.

### Pooling sensitivity falls sharply with scale

An unplanned observation, reported as exploratory since it was not
pre-registered. The spread of dip depth across the four schemes, and the
spread of the dip layer's location, differ enormously between models:

| Model | Dip-depth spread | Dip-layer spread |
|---|---|---|
| **Qwen3-8B** | **0.060** | **1 layer** |
| OLMo-2-1B | 0.086 | 2 layers |
| Qwen3-0.6B | 0.214 | 11 layers |
| bloom-1b7 | 0.199 | 15 layers |
| Salamandra-2B | 0.276 | 4 layers |

Qwen3-8B places its minimum at layer 19 under three schemes and layer 18
under the fourth, while Qwen3-0.6B, which shares its tokenizer exactly,
scatters across layers 3, 8, 12, and 14. The larger model's fingerprint is
close to pooling-invariant; the smaller model's is not. If this holds on
more models it would mean measurement stability is itself a property that
improves with scale, and it would imply that pooling caveats bind hardest
exactly where the shared space is shallowest.

## The qualitative claims

| Claim | mean | final | unit-norm | length-matched |
|---|---|---|---|---|
| BLOOM has the flattest concept space | holds | holds | holds | **does not hold** |

BLOOM is flattest under three of four schemes and is overtaken by OLMo-2-1B
only under length-matching. The claim should be reported as robust to
pooling with that single exception named.

## Fertility is measurably present in the representations

Correlating tokenizer fertility with each language's contribution to the
language variance, at each model's dip layer, across 127 languages:

| Model | Spearman | p |
|---|---|---|
| OLMo-2-1B | **+0.644** | 3e-16 |
| bloom-1b7 | +0.453 | 9e-08 |
| Qwen3-0.6B | +0.432 | 4e-07 |
| Salamandra-2B | +0.417 | 1e-06 |

Languages whose tokenizer fragments them more contribute more to the
measured language variance, in every model, at high significance. This is
the confound made visible on the representation side, where it had
previously only been controlled as a regression covariate in the downstream
validity analysis. It does not invalidate the validity results, which
already partial fertility out, but it does mean **the fingerprint itself
carries a fertility component**.

## Hypotheses

- **H1** (U-shape survives all poolings): **partially fails.** All schemes
  produce an interior minimum except BLOOM under final-token pooling, whose
  minimum sits at layer 1.
- **H2** (fertility shifts magnitude but does not explain cross-model
  differences): **supported**. The within-family control is now measured
  rather than argued. Qwen3-0.6B and Qwen3-8B share a tokenizer, so their
  fertility is identical by construction, and the scaling result that the
  larger model has the deeper shared space holds under **three of four**
  pooling schemes (0.206 vs 0.336 mean, 0.175 vs 0.276 final-token, 0.158 vs
  0.320 unit-norm), reversing only under the length-matched bound. Combined
  with final-token pooling, which has no length denominator and nearly
  preserves the full cross-model ordering, the cross-model differences are
  not a fertility artifact.
- **H3** (the identity of the most-shared layers is broadly stable):
  **fails.** Dip layers move substantially, for example Qwen3-0.6B at
  layers 8, 14, 12, and 3 under the four schemes.
- **H4** (a fertility-controlled variant retains predictive validity): not
  yet tested; requires re-running the downstream validity analysis on
  pooling variants.

## Consequences for reported claims

1. **Absolute dip-depth values are pooling-specific** and must be reported
   with the pooling scheme named. The 0.04 to 0.37 range is a mean-pooled
   quantity.
2. **The identity of the dip layer is pooling-specific**, which weakens
   layer-selection claims and is relevant wherever the LFS-selected layer is
   used to pick where other instruments are measured.
3. **Cross-model comparisons are considerably more robust than absolute
   values**, and the strongest evidence for that is the scheme with no
   length denominator reproducing the ordering exactly.
4. The within-family scaling result is untouched, since fertility is
   constant within a family by construction.
