# Aim 2 W0: offline offset and scale removal on the saved dumps (frozen 2026-09-07)

**Status.** Frozen before any run on 2026-09-07 (EDT). Zero GPU-hours; CPU partition only. This addendum freezes the W0 pre-test of the larger draft `prereg/addenda/AIM2_WORKAROUND_PROTOCOL_DRAFT_2026-09-07.md`, which remains a draft. Authorization: Waiz Khan (2026-09-07, "run the experiments on the side"); the compute rule (ask before more than about one GPU-hour) is not engaged because W0 uses no GPU.

**Question.** When the additive per-language offset and the per-language isotropic scale at a layer are removed exactly, in closed form, with no model loaded, how do the readings respond: the variance partition (LFS with components), the retrieval readings (MEXA, AaR at 1, 5, 10, 20 percent), and the scale readings (raw concept variance, mean norm, effective rank)?

**Why it matters.** The misalignment budget says 65 to 98 percent of cross-language misalignment at the dip is an additive offset plus an isotropic scale. W0 tests, without training and without a GPU, whether removing exactly that component is visible to the other reading families and whether the offsets are stable estimands. Its outcome gates the training-free forward-hook pilot W1 (about 1 GPU-hour), which needs a separate approval.

**Data.** `results/dumps/<model>/layer*.npz` on CLSP for the 14 campaign models (Qwen3-0.6B, 1.7B, 4B, 8B Base; OLMo-2-0425-1B; OLMo-2-1124-7B; Mistral-7B-v0.3; bloom-1b7; bloom-7b1; EuroLLM-1.7B; SmolLM2-1.7B; Falcon3-7B-Base; salamandra-2b; salamandra-7b), 128 languages by 1,500 sentences, fp32 pooled and fp16 stored, at the dumped layers (layer 0, the dip layer, and the other stored layers). Read-only.

**Splits.** Evaluation sentences 0 to 299 (the headline grid). Calibration sentences 300 to 1499 for the offsets. Calibration sentences 300 to 599 for the small-calibration check (P5).

**Fitted quantities, per language l on the calibration split.** mu_l = mean vector; m = mean of the 128 mu_l; b_l = mu_l - m; s_l = RMS of (x - mu_l) over the calibration sentences of l, divided by the mean RMS across languages.

**Conditions on the evaluation split.** I0 identity. I1: x - b_l. I2: m + (x - mu_l)/s_l. I3: x - r_l with r_l a random unit direction per language scaled to ||b_l||, three seeds (20260907, 20260908, 20260909). I5: x - b_l + b_eng (English swap; descriptive only).

**Readings per condition.** LFS with SS_lang, SS_con, SS_res after joint z-scoring (`src/analysis/dip_bootstrap.lfs_components`); raw concept variance L * sum_c ||mean_c - mean||^2 on unstandardized states; mean vector norm; effective rank (exponential of the spectral entropy of the covariance); mean MEXA over the 127 non-English languages against the English pivot and mean AaR at 1, 5, 10, 20 percent (`code/pilot_metrics.py` implementations copied verbatim into `src/aim2/offset_offline.py`). At the dip layer: a 300-replicate sentence bootstrap of the I1 minus I0 LFS drop (seed 20260907). Parity: the I0 LFS at layer 0 and at the dip must match `results/grid/<model>/metrics.json` within 1e-3.

**Frozen predictions (scored by `src/aim2/score_offset_offline.py`).**
- P1 (manipulation check). Under I1 at the dip, LFS falls by at least 0.10 and SS_lang by at least 50 percent in 14 of 14 models; the raw concept variance ratio I1/I0 equals 1.000 to three decimals in all (exact by construction; a deviation is a bug).
- P2 (retrieval response, both outcomes pre-labeled). Mean MEXA under I1 changes by at least 0.010 in absolute value in at least 10 of 14 models: retrieval readings respond to the additive component. Absolute change below 0.010 in at least 10 of 14: retrieval readings are near-invariant to it. Anything else: mixed, reported per model. The direction is reported, not predicted; the prior is an increase.
- P3a (scale, code check). Under I2, mean MEXA and mean AaR at 10 percent equal their I1 values to three decimals in 14 of 14 models, because cosine retrieval is invariant to a per-language scale.
- P3b (scale, moderated). Across the models with a budget file, the drop in residual share from I1 to I2 rank-correlates with the budget's median per-language scale (M2) share at Spearman at least 0.5.
- P4 (control). The mean absolute MEXA change under I3 is less than half the I1 change in at least 12 of 14 models. If I3 matches I1 within 50 percent in more models than that, the I1 retrieval effect is a magnitude artifact.
- P5 (calibration size). The I1 LFS drop with offsets fitted on 300 calibration sentences reproduces the 1,200-sentence drop within 0.02 in 14 of 14 models. Failure means the offsets are not stable at the sizes W1 would use, and W1 does not launch until they are refit.

**Deferred, not part of this freeze.** P6 (document-purged probes on centered states, about 30 CPU-hours) and P7 (share anatomy against exposure and fertility) from the draft are not run under this addendum; they need their own freeze.

**No exclusion after outcomes.** All 14 models are scored. Any failed prediction is reported as failed.

**Code.** `src/aim2/offset_offline.py` (readings), `src/aim2/score_offset_offline.py` (scoring), `slurm/aim2_w0_offset_offline.sbatch` (array of 14 on the cpu partition). Outputs: `results/aim2/offset_offline/<model>/{readings.json, manifest.json, offsets_layer<dip>.npz}` and `results/aim2/offset_offline/summary.{json,md}`. Manifests record code hash, input hashes, seed, host, and wall time.

**Execution record.** (filled after the run)

---

## Execution record (added after the run, 2026-09-07)

- CLSP array job **1787193**, cpu partition, 14 tasks, all COMPLETED (10 to 55 minutes each). Scoring: `src/aim2/score_offset_offline.py` on the cpu partition. Results synced to `results/aim2/offset_offline/`; per-model manifests carry code hash, input hashes, seed 20260907, host, wall time.
- Parity: I0 LFS at layer 0 and at the dip reproduces `results/grid` within 1e-3 for every model.

| Prediction | Result | Verdict |
|---|---|---|
| P1 LFS drop >= 0.10, SS_lang drop >= 50%, raw concept variance ratio 1.000 | 14 of 14; SS_lang drop 97 to 98 percent; ratio 1.0000 in all | PASS |
| P2 retrieval responds (abs MEXA change >= 0.010 in >= 10 of 14) | 13 of 14; MEXA rises in 12 (+0.019 to +0.076), falls in the two Salamandra models (-0.010, -0.027) | PASS, "respond" |
| P3a MEXA and AaR identical under I2 vs I1 | 0 of 14; I2 raises MEXA a further +0.02 to +0.18 | FAIL. The premise was wrong: I2 rescales around each language's own mean and then re-adds the shared mean, so cosines are not invariant to it. Recorded as a reasoning error in the frozen text, not a data problem. |
| P3b Spearman(residual-share drop I1 to I2, budget M2 share) >= 0.5 | -0.32 over 13 models | FAIL |
| P4 abs I3 change < half of abs I1 change in >= 12 of 14 | 1 of 14 | FAIL as written. The criterion ignored sign: random directions of matched norm LOWER MEXA in 12 of 14 models (-0.03 to -0.25) while the true offset RAISES it. The sign pattern supports the intended inference (the I1 effect is not a magnitude artifact), but that reading is post hoc and is labelled exploratory. |
| P5 calibration-300 reproduces calibration-1200 LFS drop within 0.02 | 8 of 14; the six failures are the shallow-dip models (SmolLM2, OLMo-2-1B, both BLOOM, EuroLLM, OLMo-2-7B; differences 0.021 to 0.035) | FAIL. Consequence per the freeze: any hook experiment must fit offsets on at least the full 1,200-sentence calibration set. |

Two of six frozen predictions passed. All 14 models are reported; nothing was excluded. Results document: `docs/AIM2_W0_RESULTS_2026-09-07.md`.
