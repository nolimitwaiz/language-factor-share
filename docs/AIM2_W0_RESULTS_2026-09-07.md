# Aim 2 W0: offline offset and scale removal on the saved dumps. Results, 2026-09-07

Protocol frozen before the run: `prereg/addenda/AIM2_W0_OFFSET_OFFLINE_PROTOCOL_2026-09-07.md` (execution record appended there). Job 1787193, CLSP cpu partition, 14 models, zero GPU-hours. Artifacts: `results/aim2/offset_offline/<model>/{readings.json, manifest.json, offsets_layer<dip>.npz}`, `summary.{json,md}`.

## What was done

At each model's dip layer, each language's additive offset b_l (its mean minus the grand mean) and its spread s_l were fitted on calibration sentences 300 to 1499 of the saved dumps and applied in closed form to the 300 evaluation sentences. Four conditions: I0 identity; I1 remove the offset; I2 remove the offset and equalize each language's spread; I3 subtract a random direction of the same norm as the offset (three seeds). Readings: LFS with components, raw concept variance, norm, effective rank, MEXA, AaR at 1, 5, 10, 20 percent. No model was loaded and nothing was trained.

## Scorecard: two of six frozen predictions passed

| Prediction | Outcome |
|---|---|
| P1 offset removal cuts LFS by >= 0.10 and SS_lang by >= 50% in all models, concept variance untouched | PASS, 14 of 14. SS_lang falls 97 to 98 percent; LFS falls to 0.03 to 0.10; raw concept variance ratio 1.0000 |
| P2 retrieval readings respond (abs MEXA change >= 0.010 in >= 10 models) | PASS, 13 of 14. MEXA rises in 12 models (+0.02 to +0.08); falls in salamandra-2b (-0.010) and salamandra-7b (-0.027) |
| P3a I2 leaves MEXA and AaR equal to I1 | FAIL, 0 of 14. Equalizing spread raises MEXA a further +0.02 to +0.18. The frozen premise (cosine invariance to a per-language scale) was wrong because I2 rescales around each language's own mean and then re-adds the shared mean |
| P3b residual-share drop under I2 tracks the budget's scale share (Spearman >= 0.5) | FAIL, Spearman -0.32 over 13 models |
| P4 random-direction control moves MEXA less than half as much as the true offset | FAIL as written, 1 of 14. The criterion ignored sign: random directions of matched norm lower MEXA in 12 of 14 models (-0.03 to -0.25) while the true offset raises it. The sign pattern supports the intended inference but is a post hoc reading, labelled exploratory |
| P5 offsets from 300 calibration sentences reproduce the 1,200-sentence LFS drop within 0.02 | FAIL, 8 of 14. The six misses are the shallow-dip models (differences 0.021 to 0.035) |

## Per-model table (dip layer)

| model | dip | LFS I0 | LFS I1 | SS_lang drop | rawCV ratio | MEXA I0 | MEXA I1 | dMEXA I1 | dMEXA I3 | I2-I1 MEXA | calib300 diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-1.7B | 15 | 0.904 | 0.083 | 97% | 1.0000 | 0.035 | 0.067 | +0.032 | -0.033 | +0.0951 | 0.0327 |
| OLMo-2-0425-1B | 9 | 0.903 | 0.088 | 97% | 1.0000 | 0.111 | 0.158 | +0.047 | +0.008 | +0.0225 | 0.0308 |
| bloom-1b7 | 14 | 0.899 | 0.096 | 97% | 1.0000 | 0.133 | 0.190 | +0.057 | -0.130 | +0.0776 | 0.0354 |
| bloom-7b1 | 17 | 0.884 | 0.088 | 97% | 1.0000 | 0.145 | 0.198 | +0.053 | -0.141 | +0.0668 | 0.0266 |
| EuroLLM-1.7B | 14 | 0.854 | 0.065 | 97% | 1.0000 | 0.187 | 0.248 | +0.062 | -0.183 | +0.0673 | 0.0239 |
| OLMo-2-1124-7B | 12 | 0.846 | 0.071 | 97% | 1.0000 | 0.186 | 0.225 | +0.039 | +0.033 | +0.0249 | 0.0211 |
| Falcon3-7B-Base | 13 | 0.826 | 0.062 | 97% | 1.0000 | 0.075 | 0.131 | +0.057 | -0.070 | +0.1119 | 0.0153 |
| salamandra-7b | 18 | 0.784 | 0.055 | 97% | 1.0000 | 0.253 | 0.225 | -0.027 | -0.250 | +0.0946 | 0.0161 |
| Mistral-7B-v0.3 | 2 | 0.744 | 0.040 | 97% | 1.0000 | 0.045 | 0.064 | +0.019 | -0.043 | +0.0907 | 0.0105 |
| salamandra-2b | 6 | 0.712 | 0.025 | 98% | 1.0000 | 0.163 | 0.153 | -0.010 | -0.161 | +0.0909 | 0.0115 |
| Qwen3-0.6B-Base | 8 | 0.728 | 0.042 | 97% | 1.0000 | 0.064 | 0.131 | +0.067 | -0.062 | +0.1441 | 0.0111 |
| Qwen3-1.7B-Base | 13 | 0.715 | 0.036 | 97% | 1.0000 | 0.081 | 0.157 | +0.076 | -0.078 | +0.1790 | 0.0081 |
| Qwen3-8B-Base | 19 | 0.608 | 0.032 | 97% | 1.0000 | 0.102 | 0.172 | +0.070 | -0.100 | +0.1580 | 0.0065 |
| Qwen3-4B-Base | 17 | 0.602 | 0.030 | 97% | 1.0000 | 0.094 | 0.158 | +0.064 | -0.092 | +0.1404 | 0.0058 |

## Reading

1. The additive per-language offset is essentially the whole language main effect at the dip: removing it deletes 97 to 98 percent of SS_lang in every model while, by construction, leaving the sentence (concept) sum of squares untouched. That is the manipulation check the training-free intervention needs.
2. The offset is visible to retrieval. Removing it raises MEXA in 12 of 14 models; the two Salamandra models are the exceptions and are reported as such. So the earlier expectation, recorded in the draft protocol, that a per-language shift is invisible to cosine retrieval was wrong and is withdrawn there too.
3. Equalizing each language's spread helps retrieval further, by more than the offset alone in most models (+0.02 to +0.18). This was not predicted. It says the isotropic-scale rung of the misalignment budget carries retrieval-relevant structure, and it is the reason the follow-on hook pilot must carry a scale condition.
4. Direction matters: a random shift of the same size hurts retrieval in 12 of 14 models. The offset direction is not an arbitrary large vector.
5. Offsets need the full 1,200-sentence calibration set for the shallow-dip models. Any hook experiment fits on all of it.

## What this does and does not license

It licenses the one-GPU-hour hook pilot (W1 in the draft protocol) with offsets fitted on 1,200 sentences and with both an offset and a scale condition, subject to Waiz and Koehn approving that GPU-hour. It does not say anything about behavior: no generation, no downstream task was run. Four of six predictions failed and are reported as failed; two failed because the frozen reasoning was wrong, which is recorded in the addendum.
