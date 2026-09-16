# Attention Mechanism for the Cross-Lingual Context Result: Scorecard

**Run 2026-07-27 against predictions frozen in
`prereg/PREREG_ATTENTION_313.md` before any attention weight was extracted.**
Qwen3-0.6B, 41 context languages, 60 sentence pairs, 28 layers, eager
attention.

## Scorecard

| Prediction | Result | Verdict |
|---|---|---|
| **P-ATT.1** matched context attracts more attention than mismatched | 41 of 41 languages; +0.0235 at L16; Wilcoxon p = 9.1e-13 | **PASS** |
| **P-ATT.2** content attention correlates with alignment | deflated rho +0.532, CI [+0.270, +0.730] | **PASS** |
| **P-ATT.3** content attention peaks in middle layers | first −0.0006, middle +0.0090, last −0.0048, peak L16 of 28 | **PASS** |
| **P-ATT.4** English context is the ceiling | English +0.0382, Spanish +0.0389 | **FAIL** |
| **P-ATT.5** content attention correlates with the behavioural benefit | deflated rho +0.293, CI [−0.093, +0.626] | **FAIL as pre-registered** |

Four of five. Both failures are informative and neither retracts the
behavioural result.

## The finding that survives: attention is where alignment shows up

The model attends preferentially to context that is *about* the target, not
merely to text in that language, in every one of 41 languages. That
attention concentrates mid-network, peaking at layer 16 of 28 (57 percent
depth), just after the depth at which the shared concept space is deepest.
And the size of that content-specific attention scales with per-language
alignment after removing data availability, language family, script, and
tokenizer cost: **rho 0.53, stable at 0.54 under a parsimonious control
set.** This is the mechanistic counterpart of the behavioural finding, and
it is the first evidence in this project that alignment corresponds to
something visible inside the computation rather than only in its outputs.

## The failure that matters, and why deflation was essential

Raw, the correlations look extraordinary: attention against alignment
0.933, attention against the behavioural benefit 0.915. Both collapse under
deflation, and the reason is visible in one number:

> **Spearman(content attention, log training tokens) = +0.919.**

How much a model attends to a foreign context is almost perfectly predicted
by how much data that language had. Reporting the raw 0.93 would have been
reporting a data-availability echo, which is precisely the failure this
project's methodology exists to prevent. Deflation removes 43 percent of the
alignment correlation and 68 percent of the behavioural one.

**P-ATT.5 is estimator-dependent and is scored as a failure.** Under the
pre-registered control set it is +0.293 with a confidence interval spanning
zero; under a parsimonious set (tokens and fertility only) it is +0.439 with
an interval excluding zero. With 38 languages and 16 design columns the full
specification leaves 22 residual degrees of freedom, so the family and
script dummies absorb a large share of the variance. The pre-registration
named the standard observable set, that set is the full one, and the result
under it fails. Both estimates are reported rather than the favourable one
chosen, following the same rule applied to the AaR-MEXA difference earlier
in this project.

**Consequence for the mechanistic claim.** Alignment predicts attention
(0.53) and alignment predicts behaviour (0.80, replicating the headline on
this subset). Attention does not independently predict behaviour once
observables are removed. So the defensible statement is that attention is a
*correlate* of alignment that is visible inside the model, not that it is
the demonstrated *channel* through which alignment produces the behavioural
benefit. Establishing the channel needs an intervention, ablating the
attention edges to the context and measuring the likelihood change, which
the pre-registration already named as the follow-up and which was not
attempted here.

## P-ATT.4, and why its failure is coherent

English was predicted to show the largest content attention as the
same-language ceiling. Spanish edged it, +0.0389 against +0.0382, a
difference of two percent. The prediction fails, but the direction is
consistent with an established result in this project: English is not the
universal best donor, and a multilingual latent factor out-predicts it for
124 of 127 languages. A model whose hub is not specifically English has no
particular reason to attend most strongly to English context.

## Scope

Single model. Attention mass is a necessary but not sufficient condition for
information to be used, and no causal claim is made. Decoder-only
architectures only.

## Artifacts

`results/attention313/Qwen3-0.6B-Base.json`, `code/attention_313.py`,
`prereg/PREREG_ATTENTION_313.md`.
