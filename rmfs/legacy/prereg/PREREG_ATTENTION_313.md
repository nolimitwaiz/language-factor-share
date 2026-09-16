# PRE-REGISTRATION: Attention mechanism for the cross-lingual context result

**Frozen 2026-07-27, before any attention weight has been extracted.**
Author: Waiz Khan.

## Motivation

The §3.1.3 experiment established, behaviorally, that per-language alignment
predicts how much a model uses a foreign-language context sentence during
generation (deflated Spearman 0.67, mismatched-context controlled). It
measured the *effect*: the target sentence becomes more predictable when a
translated context is present, beyond what any text in that language
provides.

It did not measure the *mechanism*. A likelihood improvement is consistent
with the model attending to the context, and also with several things that
are not that. This pre-registration specifies a direct test of whether the
target tokens attend to the context tokens, and whether that attention
scales with alignment.

## Quantity

For a sequence `[context tokens][separator][target tokens]` run through a
decoder-only model, for each layer:

```
raw(layer)     = mean over target positions t, mean over heads h,
                 of  sum over context positions c of  A[h, t, c]
uniform(layer) = mean over target positions t of  n_ctx / (t + 1)
ratio(layer)   = raw / uniform
```

`raw` is the share of a target token's attention mass that lands on the
context. `uniform` is what that share would be if attention were spread
evenly over all visible positions, which depends only on sequence geometry.
**`ratio` is the primary quantity**, because context and target sentences
differ in length across languages and the raw share would otherwise be
confounded with tokenizer fertility, exactly the confound this project has
been careful about elsewhere.

Attention weights are accumulated in fp32. `attn_implementation="eager"` is
required, since the default kernel does not return attention weights.

## Design

Identical to the behavioral experiment so the two are directly comparable:
the same sentence pairs, the same ~40 context languages, the same matched
and mismatched conditions. Mismatched context is the same language, a
different document. The contrast of interest is:

```
content_attention(layer) = ratio_matched(layer) - ratio_mismatched(layer)
```

which isolates attention driven by the context's *content* from attention
driven by the mere presence of text in that language.

## Predictions, frozen

**P-ATT.1** Pooled across languages, `ratio_matched > ratio_mismatched` at
the layer of maximum content attention. Direction only; a failure means the
model does not attend preferentially to relevant content and the behavioral
effect is produced by something other than attention to the context.

**P-ATT.2** (the mechanistic analog of the behavioral result) Across
languages, `content_attention` at its best layer correlates positively with
per-language alignment (MEXA at the model's best layer), Spearman > 0 at
p < 0.05, n ≈ 40. A null here means attention flow is not the channel
connecting alignment to context use, which would be a substantive negative
result about the mechanism and would not retract the behavioral finding.

**P-ATT.3** `content_attention` is larger in middle layers than in either
the first or the last layer, consistent with the shared representation
space being where cross-lingual integration happens.

**P-ATT.4** English context, the same-language ceiling, shows the largest
`content_attention` of any language.

**P-ATT.5** `content_attention` correlates positively with the behavioral
content benefit `R_content` measured in the original experiment, across
languages. This is the direct mechanism-to-effect link: languages whose
context the model attends to more should be the languages whose context
helps it more.

## Scoring

Cell by cell, reported whether passed or failed. A failure of P-ATT.2 or
P-ATT.5 is a finding about mechanism, not a retraction of the behavioral
result, and will be reported at equal prominence.

## Scope declared in advance

Decoder-only models only. Attention weights are a contested proxy for
information flow, and this experiment does not claim otherwise: it measures
where attention mass goes, which is a necessary but not sufficient condition
for the context to be used. A causal claim would require intervention
(ablating the attention edges and measuring the likelihood change), which is
not attempted here and is named as the follow-up if the correlational result
holds.
