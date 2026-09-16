# Email to Professor Koehn — FINAL, figures checked against recorded
# results. Reply in the existing thread.
# Attach: lfs_to_rmfs.pdf, rmfs_report.pdf

Subject: Re: Aim 1 update and meeting

Dear Professor Koehn,

I have two results from the Aim 1 work that I did not expect, and I
think they change how we should be reading these metrics.

To find out whether our metric would notice a broken model, I had to
build some. I trained 25 variants of Qwen3-0.6B, 75 checkpoints in all,
under different alignment objectives across languages, and I wrote down
what each metric should do before running any of them. One variant,
using the word level alignment objective from the proposal, lost about
95 percent of the structure in its representations that encodes meaning
rather than language. It then posted the best LFS score in the study.
MEXA ranked it third of the 25.

That is not a fluke of one run. LFS is a ratio, so when a model shrinks
the meaning part and the language part together, the ratio does not
move. The synthetic tests confirm it in closed form: the ratio shifts by
less than a millionth while the meaning structure falls by 99 percent.
I had written this possibility into the July preregistration along with
the detector that catches it, so the study confirmed a prediction rather
than springing a surprise.

The second result I did not predict at all. That same variant performs
normally. On a 40 language reading test it scores 0.395, against 0.390
for the untouched control. Losing nearly all of its meaning structure at
the layer we measure cost it nothing there. Meanwhile the variants that
did lose ability, around six points, look clean by every geometric
measure I have. Representational damage and capability damage are not
the same thing, and I think the field has been assuming they are.

What I built out of this is RMFS, which reports four readings per
language instead of one number. How much of the representation encodes
meaning rather than language, which is LFS's original measurement kept
intact, and which tracks how well a model carries content across
languages at 0.76. How reliably the model matches sentences to their
translations, which tracks downstream exam performance at 0.24 to 0.31,
about the same as the strongest published metric. How it behaves in its
weakest aligned languages, at 0.51 the only reading that tracked which
variants had actually lost ability. And a flag for the collapse case,
which the synthetic tests show it catches exactly. They stay separate
because each predicts something the others miss, which is the argument
against folding them back into one score. Everything converts to a 0 to
100 percentile against a reference grid of 14 models: Qwen3 (0.6B, 1.7B,
4B, 8B), OLMo-2 (1B, 7B), Mistral 7B, BLOOM (1.7B, 7.1B), EuroLLM 1.7B,
Salamandra (2B, 7B), SmolLM2 and Falcon3. The ordering comes out
sensible with no tuning: Qwen3-8B 90.5, 4B 85.7, 1.7B 73.8, the
multilingual focused Salamandra-7B 69.0, and the English centric
OLMo-2-1B 28.6.

On Aim 1 itself, 3.1.1's two questions now have quantitative answers.
The language component is measured across all 14 models, and the mapping
between languages is linear or simpler in 74 percent of the 1,792 model
and language pairs, with the nonlinear cases concentrated rather than
spread out: three models carry 80 percent of them, and in the other
eleven only 6.5 percent of languages need a nonlinear map. The 3.1.3
harness is built, validated, and supplies one of the three outcome
measures. For 3.1.4 the readings correlate with all three outcome
families after removing training data volume, language family, script
and tokenizer effects, and the same analysis puts a bound on how much
independent evidence a panel of languages actually carries, ours
included. The report lays this out per sub aim.

Two things I should flag. The 0.76 rests on four models, so its interval
runs from 0.12 to 0.84 when models rather than languages are treated as
the unit; a blind run on two untouched model families is queued to fix
that, with the predictions already recorded, and where the July walk
forward validated the earlier configuration this one tests the new form.
Separately, the 0.59 in my July email was measured on the exam capable
subset, and on the current pipeline that subset gives 0.49. I have not
yet worked out which protocol difference accounts for the gap.

Attached is a one page summary and a short report. I would like to walk
you through it if you have time this week.

Best regards,
Waiz
