# Reply to Koehn, drafted 2026-08-10 01:30 EDT
# URGENT: he offered 8am Monday Baltimore — that is this morning.
# Attach: lfs_math.pdf (and rmfs_explainer.pdf if you want the visual)

Subject: Re: Aim 1 update and meeting

Hi Professor,

8am Monday works — I will send an invite. And yes, I would like to join
the biweekly meeting on Tuesday at 8:30; thank you for the offer, it
would be useful to see what the other students are doing.

On the math: you are right that the finance framing obscured more than
it explained, and I should not have leaned on it. Attached is the
derivation with no borrowed vocabulary. The short version:

Take 300 sentences translated into 41 languages, so sentence c means
the same thing in every language. Embed all 12,300 strings at one
layer. That gives a languages by sentences grid of vectors, and for
each coordinate d we fit

    h[l,c,d] = mu_d + a[c,d] + b[l,d] + e[l,c,d]

where a[c] is an offset shared by all 41 translations of sentence c,
b[l] is an offset shared by all 300 sentences in language l, and e is
the remainder. This is the balanced two way crossed random effects
layout with one observation per cell, so the usual expected mean
squares apply:

    E[MS_L] = s2_E + N s2_L,  E[MS_C] = s2_E + L s2_C,  E[MS_E] = s2_E

giving s2_L = (MS_L - MS_E)/N and s2_C = (MS_C - MS_E)/L. Sum over
coordinates and report s2_L / (s2_L + s2_C). Zero means all
translations of a sentence land at the same point; one means all
sentences in a language land at the same point. Across 33 models it
runs 0.59 to 0.90, so language identity is the dominant organising
factor even in the best models. The attachment has a three by two
worked example with the arithmetic done out.

The same derivation shows the problem I ran into. Scale every vector by
a constant and every sum of squares scales by its square, so the ratio
is exactly unchanged while the meaning component itself can fall to
almost nothing. In the training study a checkpoint that had lost about
95 percent of its meaning variance posted the best score. That is why I
now report the components separately rather than the ratio.

One correction to how I must have presented things: I am not really
arguing that this is a better interpretability metric. The claim I can
defend is narrower — that existing measures share a specific blind spot
that I can characterise exactly, and that a validation procedure
combining controlled distortions with deliberately damaged trained
models catches it. On your point about "what the model really does"
being nebulous, I agree, which is why I stopped trying to define it and
instead correlated against three different outside measurements.

On using the metric to optimise alignment: I have evidence that this is
the dangerous direction, and it is the part I would most like your
view on. The 25 training variants include the word level alignment
objective from the proposal, and optimising it is what produced the 95
percent collapse — while every published metric I tested rated that
checkpoint highly. Separately, that damaged checkpoint performed
normally on downstream tasks, so representational damage and capability
damage came apart. My tentative conclusion is that these measures
belong in the loss as constraints rather than objectives, but I am not
confident about it.

September for ICLR is realistic and I would like to aim for it.

Best,
Waiz
