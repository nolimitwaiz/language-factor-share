# The measurement, step by step

This is the computation behind every LFS value in the paper, in the order the code performs it.
Symbols: L languages (128), N sentences (300), d coordinates, k the layer index (0 is the embedding
output, K the last block), l a language, c a sentence.

## Step 1. One vector per sentence per layer

`code/pilot_metrics.py`, function `embed_all`. Each text is tokenized and run once through the
frozen model with output of all hidden states. At each layer the token vectors are averaged over the
non-padding tokens, in fp32:

    h_lc^(k) = (1/T) * sum_t h_t^(k)

Nothing is generated. The model runs in fp16 or bf16 (fp32 for BLOOM); the pooling and everything
after it is fp32.

## Step 2. The grid

`code/cluster_grid.py` loads the first N sentences of every NTREX-128 reference and arranges the
pooled vectors as an array of shape (L, N, d) per layer. Down a column is one sentence in 128
languages; across a row is one language over 300 sentences.

## Step 3. Joint standardization

Each coordinate is standardized over all L x N cells of the grid: subtract its mean, divide by its
standard deviation (plus 1e-9). This is done jointly, never per language, because per-language
standardization would remove the language differences being measured. It matters: in Qwen3-8B one
coordinate carries 92.8 percent of the raw spread at layer 19, and the share would read 0.347 instead
of 0.608 without this step (`docs/study/ch8_numbers.tex`).

## Step 4. Three sums of squares and the share

`code/pilot_metrics.py`, function `lfs`, and `src/analysis/dip_bootstrap.py`, function
`lfs_components` (identical arithmetic, the second also returns the components):

    z_l = (1/N) sum_c z_lc         language mean
    z_c = (1/L) sum_l z_lc         sentence mean
    z   = (1/LN) sum_lc z_lc       grand mean
    SS_lang = N sum_l ||z_l - z||^2
    SS_con  = L sum_c ||z_c - z||^2
    SS_res  = sum_lc ||z_lc - z_l - z_c + z||^2
    LFS = SS_lang / (SS_lang + SS_con)

On a balanced grid the three sums add exactly to the total centered sum of squares. The residual is
excluded from the ratio on purpose, so that a noisy model does not look language-neutral because its
residual is large. Under an independent Gaussian null the two sums of squares are independent
chi-square variables, so LFS is Beta-distributed with expectation exactly (L-1)/(L+N-2): 0.298 at
(128, 300). Empirical standardization makes this approximate; `src/analysis/estimator_nulls.py` and
`tests/test_estimator_floor.py` check it numerically.

Stored per layer in `results/grid/<model>/metrics.json` under `per_layer[k]["lfs"]`: `lfs`,
`var_lang`, `var_concept`, `var_resid` (the three sums as fractions of the total).

## Step 5. The depth profile and its summary

Steps 1 to 4 are repeated at every layer. Three numbers summarize a profile: the minimum layer
k* = argmin_k LFS^(k), the minimum LFS, and the dip depth LFS^(0) - LFS^(k*). A profile "dips and
recovers" when the minimum is interior and both endpoints are higher.
`rmfs/scripts/audit_lfs_headline_claims.py` computes these from the stored profiles and writes
`rmfs/results/tables/lfs_depth_profile_audit.csv` and `lfs_headline_audit.json`.

## Step 6. Companion measures on the same grid

`code/cluster_grid.py` also computes, per layer and language against the English pivot:

* **MEXA** (Kargaran et al., 2024), reimplemented under the same preprocessing: the fraction of
  sentences whose translation is the nearest neighbor; reported as a reference measure.
* **AaR@10**, alignment-at-risk: the mean retrieval margin of the worst decile of sentences.

and, at the best (most shared) layer, the **hub test** (`hub_with_controls`): for each target
language, the cross-validated R^2 of predicting its vectors from English versus from the average of
the other languages (excluding English and the target). Representations are reduced to 64 principal
components fitted on the full grid, predictors are standardized on the full sentence set, and ridge
regression (penalty 1) is fitted with three-fold cross-validation over sentences; a fixed panel of 12
single-language donors is also ranked.

## Step 7. The variance-component estimator, LFS-VC

`rmfs/src/rmfs/components/variance.py`. The same grid is treated as a balanced two-way random-effects
layout with one observation per cell. Per coordinate, with mean squares MS_L, MS_C, MS_E and their
degrees of freedom, the interior estimates are

    sigma_L^2 = (MS_L - MS_E) / N,     sigma_C^2 = (MS_C - MS_E) / L

Non-negativity is enforced by refitting on the boundary: a negative component is set to zero, the
error is re-pooled with that effect's sum of squares, and the other component is re-estimated. When
both initial estimates are negative the implementation sets both to zero, which is not always the
constrained optimum; in every saved fit (104 records under `rmfs/results/runs/`) no coordinate
produced a negative estimate, so this branch never affected a reported value. Summing over
coordinates, LFS-VC = sigma_L^2 / (sigma_L^2 + sigma_C^2), and the sentence-component share used in
the behavioral analysis is 1 - LFS-VC at the dip layer. The two estimators are distinct: on the 14
models measured by both, the sentence-component shares at the saved dip layer agree at Spearman
1.000 (maximum absolute difference 0.005), but the levels differ.

## Step 8. Everything downstream

The behavioral tests, the stress tests, and the training interventions all start from these stored
grids or dumps and are described in `docs/04_experiments.md` and `docs/05_training_interventions.md`.
Each ran under a protocol frozen before the run (`prereg/`), and every failed prediction is reported.

## A worked example you can do by hand

Two languages, three sentences, one coordinate:

    English: 2  6  10        language means 6 and 9
    Arabic:  5  9  13        sentence means 3.5, 7.5, 11.5; grand mean 7.5

    SS_lang = 3 * [(6-7.5)^2 + (9-7.5)^2] = 13.5
    SS_con  = 2 * [(3.5-7.5)^2 + 0 + (11.5-7.5)^2] = 64
    SS_res  = 0 (this grid is exactly additive)
    LFS = 13.5 / 77.5 = 0.174

For Qwen3-8B at layer 19 on the real grid: SS_lang 55.2 million, SS_con 35.6 million, residual
66.5 million, so LFS = 0.608, against 0.945 at layer 0 and 0.958 at layer 36.
