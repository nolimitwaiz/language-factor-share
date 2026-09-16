# The experiments

One section per experiment: the question, the setup with sizes, the command that produced it, where
the result is stored, and what it showed. Numbers are the ones in the paper; the artifact behind each
is listed in `docs/06_results_map.md`.

## 1. Depth profiles on 17 primary models

**Question.** Does the language share dip at an interior layer and recover in every model?

**Setup.** 17 open base models from 11 families (`docs/02_models.md`), the 128 x 300 NTREX grid,
direct sums of squares at every layer. Ten models formed the development grid; SmolLM2-1.7B and
Falcon3-7B were run afterwards under predictions written and frozen first;
Salamandra, granite, Yi, and Llama were added under frozen predictions after that.

**Command.** `python code/cluster_grid.py --model_idx $SLURM_ARRAY_TASK_ID` (`cluster/submit_grid.sh`,
`cluster/submit_grid3.sh`, `cluster/submit_big1.sh`).

**Result.** `results/grid/<model>/metrics.json`; summary `rmfs/results/tables/lfs_depth_profile_audit.csv`.
All 17 profiles have an interior minimum with both endpoints higher. Dip depth runs from 0.039
(BLOOM-1.7B) to 0.336 (Qwen3-8B). Within the Qwen3 series it grows with size (0.206, 0.213, 0.314,
0.336); across families at similar size it varies more than within a family, so training recipe
matters more than parameter count. This comparison is descriptive.

## 2. Expansion set: 19 more models with LFS-VC

**Setup.** 19 models from 13 families at pinned revisions, measured with the variance-component
estimator under the content-transfer protocol. **Result.** 19 of 19 dip and recover; three minima
fall below 0.5 under LFS-VC (Mistral-Nemo 0.377, mGPT 0.444, Llama-3.1-8B 0.479, which is 0.626 under
direct sums of squares on the primary grid). The two estimators are never pooled.

## 3. Second corpus, resampled sentences, instruction-tuned pairs

Protocol: `prereg/addenda/AIM1_EABC_PROTOCOL_2026-09-05.md` (twelve predictions frozen before the
runs). Scoring: `src/analysis/score_eabc.py`; write-up `docs/AIM1_EABC_RESULTS_2026-09-05.md`.

* **FLORES-200 replication (E-B).** The same 17 models on 116 languages, first 300 devtest sentences
  (`code/cluster_grid_corpus.py`, job `slurm/e_b_flores_grid.sbatch`). 17 of 17 dip and recover; dip depths
  correlate with NTREX at Spearman 0.92, median absolute difference 0.011; granite is the outlier
  (0.190 on NTREX, 0.074 on FLORES). `results/flores_grid/`.
* **Sentence resampling (E-A).** For the 14 models with 1,500-sentence dumps, 2,000 subsets of 300
  sentences drawn without replacement, same index sets for every model, at the stored layers
  (`src/analysis/dip_bootstrap.py`, job `slurm/e_a_dip_bootstrap.sbatch`). 95 percent half-widths 0.004 to 0.020 for 12 models; 0.045
  (Mistral-7B) and 0.059 (Salamandra-2B), the one failed prediction of twelve (bound 0.03). The
  model ordering agrees with the full grid at Spearman at least 0.90 in 2,000 of 2,000 subsets
  (median 0.996). `results/dip_bootstrap/`.
* **Instruction tuning (E-C).** Six base and instruction-tuned pairs on the same grid (job `slurm/e_c_instruct_grid.sbatch`). Dip depth moves
  by at most 0.01; the minimum layer agrees within two layers for five of six. `results/instruct_grid/`.

## 4. Is the shared middle English? The hub test

**Setup.** At each model's dip layer, for each of the 127 non-English languages, predict its vectors
from English or from the average of the other languages (excluding English and the target); ridge on
64 principal components, three-fold cross-validation over sentences (`hub_with_controls` in
`code/cluster_grid.py`; details in `docs/03_pipeline.md`, step 6).

**Result.** The multilingual average wins for 113 (BLOOM-1.7B) to 124 of 127 languages depending on
the model, by a median +0.06 to +0.16 R^2. Among the 12 tested donors, English is the best for 9 of
10 high-resource targets but 1 of 8 low-resource ones. `rmfs/results/tables/lfs_hub_audit.csv`.

## 5. How complicated is the difference between languages? The misalignment decomposition

**Setup.** `src/analysis/budget_real.py` (job `slurm/campaign_budget.sbatch`) on the 1,500-sentence dumps of 14 models at the dip layer,
common rank 64. Each language is mapped into a shared reference space (a generalized-Procrustes
consensus) by a nested sequence of maps: per-language offset (M1), isotropic scale (M2), rotation
(M3), general linear (M4), nonlinear kernel (M5). Each map is credited with the reduction in held-out
reconstruction error it adds, over 20 document-purged, length-stratified splits, against
per-language permutation nulls. Shares are fractions of the initial held-out mapping error.

**Result.** Offset plus scale remove 63 to 97 percent of the misalignment (Qwen3-4B 63 percent,
OLMo-2-1B 97 percent); rotation never exceeds 0.7 percent; a linear or simpler map is selected for
74 percent of the 1,792 model-language pairs. A document-purged re-audit on Qwen3-1.7B lowered the
nonlinear share by six points, so nonlinear shares may be inflated by leakage.
`results/budget/*_fixed_raw/budget.json`; figure `code/report_figures4.py`.

## 6. Does the model use context given in another language? Content transfer

Protocol frozen 2026-08-24: `prereg/addenda/AIM1_R1_CONTENT_TRANSFER_PROTOCOL_2026-08-24.md`.

**Setup.** 100 document-distinct NTREX pairs (an English sentence and the preceding sentence of the
same story), context in 41 languages, 19 models, 157,700 scored sequences. The benefit of a context
is the loss on the English target under a length-matched mismatched context minus the loss under the
true context, normalized by the same benefit with English context (R_content). Scripts:
`src/analysis/build_r1_content_pairs.py`, `src/analysis/r1_content_transfer.py`,
`src/analysis/assemble_analyze_r1_content_transfer.py`; jobs `slurm/r1_content_transfer_full.sbatch` and `slurm/r1_content_transfer_analyze.sbatch` (the analysis runs only after every array element has
terminated and validates every sealed artifact first).

**Result.** Models with a smaller language share at the dip layer (larger 1 - LFS-VC) use the
foreign context more: Spearman 0.77 [0.44, 0.94] at the model level, 0.51 [0.19, 0.71] after
adjusting for the exposure proxy, language family, script, and tokenizer fertility (model bootstrap,
646 model-language rows), positive under leave-one-family-out. MEXA on the same test: 0.59
[0.33, 0.82], reported for reference. `results/round3/r1_content_transfer_analysis_2026-08-24/`.

## 7. Pooled benchmark accuracy after covariate adjustment

**Setup.** `code/attribution.py` fits, for each benchmark outcome on the logit scale above chance,
model fixed effects plus log10 CulturaX tokens, Glottolog macro-family, script, and tokenizer
fertility, with cluster-robust errors by language; the residual is the adjusted outcome, and validity
is the Spearman correlation between the residualized measure and the residualized outcome with
cluster bootstraps by language, by model, and crossed. On the 17-model grid (1,241 rows) the exposure
coefficient is +0.286 (SE 0.059) and R^2 = 0.824. The expanded 33-model panel against pooled Belebele
and INCLUDE accuracy is `rmfs/scripts/expanded_validity.py`.

**Result.** 1 - LFS-VC: 0.04 [-0.05, 0.12] on 1,740 rows over 70 languages; MEXA 0.24; AaR 0.10.
No clear association; reported as a limit of LFS as a performance indicator. Two independent
benchmarks on the same cells have adjusted residuals that correlate at only 0.315, so the criterion
itself is noisy. `rmfs/results/tables/expanded_validity.txt`, `results/deflation/attribution/`.

## 8. What the measures can and cannot see: fault injection and invariance

Protocol: `prereg/PREREG_BATTERY.md` (frozen 2026-07-15), `prereg/addenda/ROUND3_EXECUTION_PROTOCOL_2026-08-24.md`.

**Setup.** Known faults are injected into saved representations: per-language shifts, rotations of
0.1 to 0.8 radians, uniform shrinking, and shrinking of the non-English languages toward the global
centroid by 30, 60, or 90 percent (`src/battery/`, `src/analysis/round3_invariance.py`, job `slurm/round3_invariance.sbatch`), and every
measure is scored against predictions frozen in advance.

**Result.** The decomposition recovers shifts and rotations. Uniform shrinking changes nothing in
LFS, MEXA, or CKA because all three are scale-invariant; only the content-variance preservation
check (CVP, the within-language variance across sentences relative to the reference) responds,
returning (1 - m)^2 to three decimals for shrinkage m. In the 14-model analytic sweep, 3,414 of
3,416 invariance checks pass; the two exceptions are numerical-tolerance flags below 1e-7 on AaR and
the mean margin for one model. `results/battery/`, `results/round3/invariance_core_2026-08-24/`.

## 9. Measurement dependence

Pooling choice (mean, final token, unit norm, length matched) moves absolute dip depth by 0.13 to
0.20 while the cross-model ordering changes by at most one adjacent transposition
(`src/analysis/pooling_variants.py`, `results/pooling_variants/`). ZCA whitening changes the ordering
substantially (Spearman 0.45; `results/whitened/`). Tokenizer fertility correlates with each
language's contribution to SS_lang (`docs/FERTILITY_POOLING_SCORECARD.md`). Rotating an already
standardized grid leaves LFS unchanged; rotating raw states and standardizing again does not
(`src/analysis/axis_basis_sensitivity.py`).

## 10. A language probe, attention, and the logit lens

A linear probe on Qwen3-0.6B (24 languages, 29 layers, split by sentence) recovers language
identity at 0.875 to 0.886 accuracy (chance 0.042) at every layer, including the layer with the
smallest share (job `slurm/adv_probe.sbatch`): variance share and separability are different quantities
(`results/adversarial_probe/`). Matched context draws more attention than mismatched context in 41
of 41 languages (`code/attention_313.py`, `results/attention313/`). Mid-stack states reach a maximum
cosine of about 0.10 to any output embedding, so logit-lens readouts at the most shared layers are
weakly supported by the geometry (`code/lens_312.py`, `results/lens312/`).

## 11. Offline offset removal (pre-test for the next aim)

Protocol frozen 2026-09-07: `prereg/addenda/AIM2_W0_OFFSET_OFFLINE_PROTOCOL_2026-09-07.md`.
`src/aim2/offset_offline.py` (job `slurm/aim2_w0_offset_offline.sbatch`) removes the per-language offset (and optionally a scale) from the saved
dumps with no training. Removing the offset deletes 97 to 98 percent of SS_lang in every model while leaving the sentence sum of squares untouched; two of six frozen
predictions passed; MEXA rose in 12 of 14 models and fell for the two Salamandra models.
`results/aim2/offset_offline/`, `docs/AIM2_W0_RESULTS_2026-09-07.md`.
