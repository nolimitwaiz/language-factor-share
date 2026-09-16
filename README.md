# Language-Factor Share (LFS)

Code, frozen protocols, stored results, figures, and documents behind the paper
**"Measuring the Language Share of Multilingual Representations"** (Waiz Khan and Philipp Koehn,
Johns Hopkins University; submitted to ICLR 2027). The work is Aim 1 of the NSF project
*Understanding and Improving Multilinguality of Large Language Models*.

Everything a reader needs to check a number in the paper is here: the script that produced it,
the stored result it was read from, and the protocol that was frozen before the run. Hidden-state
dumps and model checkpoints are not stored in the repository; they live on the compute cluster
and can be regenerated with the scripts in `slurm/` and `cluster/`.

## The question and the measurement in one paragraph

A multilingual language model turns a sentence into a vector at every layer. How much of that
vector is about the *language* the sentence is written in, and how much about the *sentence itself*?
LFS answers this with a two-way analysis of variance. Take the same N sentences translated into L
languages, run each translation once through a frozen model, mean-pool the hidden states at one
layer, and arrange the vectors as a grid with one row per language and one column per sentence.
After standardizing every coordinate jointly over the grid, the total spread splits exactly into a
language part, a sentence part, and a residual:

    SS_lang = N * sum_l ||z_l - z||^2        (how far the language means sit from the grand mean)
    SS_con  = L * sum_c ||z_c - z||^2        (how far the sentence means sit from the grand mean)
    SS_res  = sum_{l,c} ||z_lc - z_l - z_c + z||^2

    LFS = SS_lang / (SS_lang + SS_con)       defined when SS_lang + SS_con > 0

LFS is 1 when only language separates the vectors and 0 when only the sentence does. Under pure
noise it is not zero but (L-1)/(L+N-2), which is 0.298 on the 128-language, 300-sentence grid used
throughout. Repeating the computation at every layer gives a *depth profile*.

## What the paper found

| Result | Value | Where |
|---|---|---|
| Every depth profile has an interior minimum with both endpoints higher | 17 of 17 primary models (direct sums of squares), 19 of 19 expansion models (LFS-VC) | `results/grid/`, `rmfs/results/tables/lfs_headline_audit.json` |
| Dip depth (layer-0 LFS minus the minimum) ranges nine-fold | 0.039 (BLOOM-1.7B) to 0.336 (Qwen3-8B) | `rmfs/results/tables/lfs_depth_profile_audit.csv` |
| The profile replicates on a second corpus | FLORES-200: 17 of 17, depth ordering Spearman 0.92 | `results/flores_grid/`, `results/eabc_scoring/` |
| The shared middle is not English | the average of the other languages predicts a language better than English for 113 to 124 of 127 languages | `results/grid/*/metrics.json` (`hub_at_best_layer`) |
| The difference between languages is mostly an offset plus a scale | 63 to 97 percent of held-out cross-language misalignment | `results/budget/*_fixed_raw/budget.json` |
| A smaller language share goes with more use of foreign-language context | Spearman 0.51 [0.19, 0.71] after covariate adjustment, 19 models | `results/round3/r1_content_transfer_analysis_2026-08-24/` |
| No clear association with pooled benchmark accuracy | 0.04 [-0.05, 0.12], 33 models | `rmfs/results/tables/expanded_validity.txt` |
| A share cannot see uniform shrinking | norm 451 to 42, raw content variance 13,791 to 699, LFS 0.445 to 0.493 | `results/aim2/evaluation_wordalign.json` |
| LFS is not a training objective | objectives that pushed LFS down pushed tail alignment down with it (Pearson +0.92 over 18 condition-seed points) | `results/aim2/evaluation.json` |

The full mapping from every number in the paper to its artifact and code is in
[`docs/06_results_map.md`](docs/06_results_map.md).

## The pipeline, step by step

| Step | What happens | Code | Output |
|---|---|---|---|
| 1 | Load the parallel corpus: the first 300 sentences of NTREX-128 in all 128 languages | `code/cluster_grid.py` (`load_ntrex`), `data/NTREX/` | in memory |
| 2 | Run each translation once through a frozen model; at every layer, mean-pool the hidden states over non-padding tokens in fp32 | `code/pilot_metrics.py` (`embed_all`) | one vector per (language, sentence, layer) |
| 3 | Standardize each coordinate jointly over all L x N cells (never per language) | `code/pilot_metrics.py` (`lfs`) | standardized grid |
| 4 | Compute the language, sentence, and residual sums of squares and the share, at every layer | `code/pilot_metrics.py` (`lfs`); `src/analysis/dip_bootstrap.py` (`lfs_components`) | `results/grid/<model>/metrics.json` |
| 5 | Summarize the depth profile: minimum layer, minimum LFS, dip depth | `rmfs/scripts/audit_lfs_headline_claims.py` | `rmfs/results/tables/lfs_depth_profile_audit.csv` |
| 6 | Compute the companion measures on the same grid: MEXA and AaR@10 per language against the English pivot, the English-versus-multilingual hub test at the best layer | `code/cluster_grid.py` | same `metrics.json` |
| 7 | The variance-component estimator LFS-VC (REML, non-negative) for the expansion set and the behavioral analysis | `rmfs/src/rmfs/components/variance.py` | `rmfs/results/runs/*/result.json` |
| 8 | Behavioral tests, stress tests, and training interventions, each under a protocol frozen before the run | `src/analysis/`, `src/aim2/`, `src/battery/`, `prereg/` | `results/`, `docs/*_RESULTS_*.md` |

Details with commands: [`docs/03_pipeline.md`](docs/03_pipeline.md).

## The experiments

| Experiment | Question | Protocol | Code | Result | Write-up |
|---|---|---|---|---|---|
| Depth profiles, 17 primary models | Does every model dip and recover? | `prereg/`, predictions frozen before the SmolLM2 and Falcon3 runs | `code/cluster_grid.py` | `results/grid/` | `docs/AIM1_METRICS_MODELS_DATASETS_EXPERIMENTS_2026-08-24.md` |
| Expansion set, 19 models | Does the profile hold beyond the primary families? | `prereg/addenda/AIM1_R1_CONTENT_TRANSFER_PROTOCOL_2026-08-24.md` | `rmfs/scripts/new_model_readings.py`, `rmfs/scripts/layer_profiles.py` | `rmfs/results/` | `docs/AIM1_R1_CONTENT_TRANSFER_RESULTS_2026-08-26.md` |
| Second corpus (FLORES-200), resampled sentences, instruction-tuned pairs | Does the profile replicate, how uncertain is it, does instruction tuning change it? | `prereg/addenda/AIM1_EABC_PROTOCOL_2026-09-05.md` | `code/cluster_grid_corpus.py`, `src/analysis/dip_bootstrap.py`, `src/analysis/score_eabc.py` | `results/flores_grid/`, `results/dip_bootstrap/`, `results/instruct_grid/`, `results/eabc_scoring/` | `docs/AIM1_EABC_RESULTS_2026-09-05.md` |
| English-versus-multilingual hub test | Is the shared middle English? | in `code/cluster_grid.py` docstring | `code/cluster_grid.py` (`hub_with_controls`) | `results/grid/*/metrics.json`, `rmfs/results/tables/lfs_hub_audit.csv` | technical report, hub section |
| Misalignment decomposition | How complicated is the map between languages? | `prereg/PREREG_BATTERY.md` (part 2) | `src/analysis/budget_real.py` | `results/budget/` | technical report, decomposition section |
| Content transfer (foreign-language context) | Does the model use context given in another language, and does LFS track it? | `prereg/addenda/AIM1_R1_CONTENT_TRANSFER_PROTOCOL_2026-08-24.md` | `src/analysis/build_r1_content_pairs.py`, `src/analysis/r1_content_transfer.py`, `src/analysis/assemble_analyze_r1_content_transfer.py` | `results/round3/r1_content_*` | `docs/AIM1_R1_CONTENT_TRANSFER_RESULTS_2026-08-26.md` |
| Pooled benchmark accuracy with covariate adjustment | Does LFS predict Belebele and INCLUDE accuracy once training-data exposure is adjusted for? | `code/attribution.py` docstring | `code/attribution.py`, `rmfs/scripts/expanded_validity.py` | `results/deflation/`, `rmfs/results/tables/expanded_validity.txt` | `docs/RMFS_ADDITIONAL_VALIDATION_RESULTS_2026-08-16.md` |
| Invariance and fault-injection suite | What can the measures see, and what are they blind to? | `prereg/PREREG_BATTERY.md`, `prereg/addenda/ROUND3_EXECUTION_PROTOCOL_2026-08-24.md` | `src/battery/`, `src/analysis/round3_invariance.py` | `results/battery/`, `results/round3/invariance_core_2026-08-24/` | `docs/ROUND3_PHASE_A_CORE_RESULTS_2026-08-24.md` |
| Measurement dependence | Do pooling, whitening, tokenizer fertility change the ordering? | `prereg/addenda/AIM1_AXIS_BASIS_SENSITIVITY_PROTOCOL_2026-08-24.md` | `src/analysis/pooling_variants.py`, `src/analysis/tail_whiten.py`, `src/analysis/axis_basis_sensitivity.py` | `results/pooling_variants/`, `results/whitened/`, `results/round3/axis_basis_sensitivity_2026-08-24/` | `docs/FERTILITY_POOLING_SCORECARD.md`, `docs/AIM1_AXIS_BASIS_SENSITIVITY_RESULTS_2026-08-24.md` |
| Training interventions (Aim 2 studies) | What happens when LFS or alignment is used as a training signal? | `prereg/PREREG_WORDALIGN.md`, `prereg/PREREG_CODESWITCH.md`, `docs/AIM2_STUDY_PLAN.md` | `src/aim2/` | `results/aim2/` | `docs/AIM2_STUDY_SCORECARD.md`, `docs/CODESWITCH_SCORECARD.md`, `docs/WORDLEVEL_STAGE1_SCORECARD.md` |
| Offline offset removal (pre-test for Aim 2) | How much of the language part is the per-language offset? | `prereg/addenda/AIM2_W0_OFFSET_OFFLINE_PROTOCOL_2026-09-07.md` | `src/aim2/offset_offline.py` | `results/aim2/offset_offline/` | `docs/AIM2_W0_RESULTS_2026-09-07.md` |

Each experiment is described in plain terms, with sizes and commands, in
[`docs/04_experiments.md`](docs/04_experiments.md) and
[`docs/05_training_interventions.md`](docs/05_training_interventions.md).

## Repository map

```
code/            extraction, per-layer measures, hub test, benchmark adjustment, report figures
src/analysis/    frozen-protocol analyses: resampling, replication scoring, content transfer, decomposition, invariance
src/aim2/        training interventions and their evaluation
src/battery/     synthetic fault injection on saved representations
src/campaign/    the 1,500-sentence hidden-state dumps used by the decomposition and resampling
rmfs/            the stress-test program: LFS-VC estimator, expansion-set profiles, benchmark panels, ledgers, tests
prereg/          protocols frozen before each run, with dated addenda
results/         stored results (JSON, CSV, parquet); no hidden states or checkpoints
manifests/       model revision locks and job manifests
ledgers/         claims ledger (paper number -> artifact -> code) and the cluster run ledger
slurm/, cluster/ the job scripts that produced every model-touching result
tests/           unit tests for the estimators and analyses
paper/           the ICLR source, the technical report, the findings brief, figures and tables
docs/            guides (01 to 08) and the dated results write-ups and scorecards
docs/study/      the study guide and its figures
presentations/   slide decks and the scripts that build them
data/            small metadata files and the language mapping; corpora are downloaded, see docs/01_data.md
```

## Reproducing

* Reading and re-deriving every table and figure from the stored results needs only a CPU and
  `pip install -r requirements.txt`. See [`docs/08_reproduce.md`](docs/08_reproduce.md).
* Regenerating the stored results needs the models: the `slurm/` and `cluster/` scripts ran on a
  SLURM cluster with single GPUs (RTX 2080 Ti or A100). Inference precision is fp16 or bf16;
  pooling and all statistics are fp32. BLOOM is never run in fp16.
* Every discrepancy found during the project, with its cause and fix, is in
  [`docs/DISCREPANCY_LEDGER.md`](docs/DISCREPANCY_LEDGER.md) (summarized in
  [`docs/07_corrections.md`](docs/07_corrections.md)).

## Guides

1. [`docs/01_data.md`](docs/01_data.md): corpora, benchmarks, covariates, sizes, licenses
2. [`docs/02_models.md`](docs/02_models.md): the 17 primary models, the 19 expansion models, the instruction-tuned pairs, the training model
3. [`docs/03_pipeline.md`](docs/03_pipeline.md): the measurement, step by step, with the formulas and the code
4. [`docs/04_experiments.md`](docs/04_experiments.md): each experiment, what it asks, how it was run, what it found
5. [`docs/05_training_interventions.md`](docs/05_training_interventions.md): the training conditions, sizes, and outcomes
6. [`docs/06_results_map.md`](docs/06_results_map.md): every number in the paper, its artifact and code
7. [`docs/07_corrections.md`](docs/07_corrections.md): the corrections ledger in brief
8. [`docs/08_reproduce.md`](docs/08_reproduce.md): environment, commands, and job scripts

## Citation

See `CITATION.cff`. Code is released under the MIT license; datasets keep their own licenses.

## Contact

Waiz Khan, Department of Applied Mathematics and Statistics, Johns Hopkins University.
