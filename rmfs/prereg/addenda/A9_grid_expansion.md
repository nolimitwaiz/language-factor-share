# Addendum A9 — expanding the reference grid from 14 to 35 models

**Dated 2026-08-08, written BEFORE any new model is run. The project's
own analysis identified the model dimension as the binding constraint:
the content transfer result carries [0.60, 0.89] when language families
are the resampling unit but [0.12, 0.84] when models are, and the
percentile calibration quantizes visibly at 14 reference models. This
addendum expands the grid and states what is expected to change.**

## What is added

21 models, chosen for scale coverage and for diversity of multilingual
design (dedicated multilingual pretraining, European focused, Asian
focused, and English centric), all reachable without new licence
acceptance:

Qwen2.5 (0.5B, 1.5B, 3B, 7B); XGLM (1.7B, 2.9B, 7.5B); mGPT;
OLMo-2-1124-13B; granite-3.1-8b-base; Yi-1.5-9B; TowerBase-7B;
occiglot-7b-eu5; internlm2_5-7b; Sailor (1.8B, 7B); EuroLLM-9B;
Llama-3.1-8B; SmolLM2-360M; Falcon3-3B-Base; Mistral-Nemo-Base-2407.

**Excluded on principle:** gemma-2-9b and aya-23-8B, although both are
reachable. The blind walk-forward families are Gemma and Aya; running
any sibling model would contaminate that test. This exclusion costs two
otherwise useful models and is recorded here so the choice is visible.

## Protocol

Identical to the existing 14 in every respect. The measurement layer
for a new model is selected by the same rule already applied to the
grid: the layer at which the language share is at its minimum,
determined from a sweep over all layers on the same 300 sentences and
41 languages. No per model tuning of anything else.

Downstream targets: both exams, translated and native, scored zero shot
by log likelihood exactly as before.

## What changes, disclosed in advance

Percentile calibration is defined against the reference grid, so
expanding the grid **re-issues the scores**. Any score quoted after
this date is calibrated on 35 models and is not comparable digit for
digit with a score quoted before it. Both calibrations are kept; the
grid version is recorded beside every score from here on.

## Frozen predictions

- **G1 (shape).** The U shaped depth profile reproduces: the language
  share is highest at the embedding and the output, with its minimum in
  the middle to late layers, in at least 18 of the 21 new models.
- **G2 (the reason for doing this).** With models as the resampling
  unit, the interval on the content transfer result is narrower than
  the current [0.12, 0.84]; specifically its width falls by at least
  25 percent.
- **G3 (validity holds).** On the full language panel, the matching
  reading's deflated correlation with the pooled exam target remains
  positive with the 95 percent interval excluding zero.
- **G4 (recalibration is modest).** Re-issued percentile scores for the
  original 14 models move by no more than 10 points on average. A
  larger move is reported as a finding about the calibration's
  stability.

G2 is the point of the exercise. If it fails, adding models did not buy
precision and the limitation stands as previously reported.
