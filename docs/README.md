# Documentation index

Start with the guides, in order. The dated write-ups are the primary records of each study,
written when the study finished; the guides summarize them and point back to them.

## Guides

* [`00_research_rules.md`](00_research_rules.md): the research and compute rules every run followed; measure boundaries
* [`01_data.md`](01_data.md): corpora, benchmarks, covariates, subset sizes, licenses, what is not redistributed
* [`02_models.md`](02_models.md): the 17 primary models, the 19 expansion models, the instruction-tuned pairs, the training model
* [`03_pipeline.md`](03_pipeline.md): the measurement step by step, with formulas, a hand-computable example, and the code behind each step
* [`04_experiments.md`](04_experiments.md): each experiment: question, sizes, command, job script, output path, result
* [`05_training_interventions.md`](05_training_interventions.md): every training condition, its objective, sizes, and outcome
* [`06_results_map.md`](06_results_map.md): every number in the paper mapped to its artifact and the script that produced it
* [`07_corrections.md`](07_corrections.md): the discrepancy ledger in brief
* [`08_reproduce.md`](08_reproduce.md): environment, commands, job scripts, and tests, in run order

## Results write-ups and scorecards

**Aim 1: depth profiles and their audits**

* [`AIM1_METRICS_MODELS_DATASETS_EXPERIMENTS_2026-08-24.md`](AIM1_METRICS_MODELS_DATASETS_EXPERIMENTS_2026-08-24.md): Aim 1 Metric, Dataset, Model, and Experiment Registry
* [`AIM1_PROFILE_AXES_RESULTS_2026-08-24.md`](AIM1_PROFILE_AXES_RESULTS_2026-08-24.md): Aim 1 Profile 0.2 and Coordinate-Axis Results
* [`AIM1_AXIS_BASIS_SENSITIVITY_RESULTS_2026-08-24.md`](AIM1_AXIS_BASIS_SENSITIVITY_RESULTS_2026-08-24.md): Aim 1 Orthogonal-Basis Sensitivity Results

**Aim 1: replication, resampling, instruction tuning**

* [`AIM1_EABC_RESULTS_2026-09-05.md`](AIM1_EABC_RESULTS_2026-09-05.md): Aim 1 E-A / E-B / E-C results: dip-depth bootstrap, FLORES-200 replication, instruct versus base

**Aim 1: content transfer (foreign-language context)**

* [`AIM1_R1_CONTENT_TRANSFER_RESULTS_2026-08-26.md`](AIM1_R1_CONTENT_TRANSFER_RESULTS_2026-08-26.md): Aim 1 R1: 19-Model Content-Transfer Expansion

**Stress tests: invariance, attention, pooling, fertility**

* [`ROUND3_PHASE_A_CORE_RESULTS_2026-08-24.md`](ROUND3_PHASE_A_CORE_RESULTS_2026-08-24.md): Round 3 Phase A Core Results
* [`ATTENTION_313_SCORECARD.md`](ATTENTION_313_SCORECARD.md): Attention Mechanism for the Cross-Lingual Context Result: Scorecard
* [`FERTILITY_POOLING_SCORECARD.md`](FERTILITY_POOLING_SCORECARD.md): Fertility and Pooling Robustness: Scorecard

**Benchmark validity**

* [`RMFS_ADDITIONAL_VALIDATION_RESULTS_2026-08-16.md`](RMFS_ADDITIONAL_VALIDATION_RESULTS_2026-08-16.md): RMFS additional validation results, 2026-08-16

**Training interventions (Aim 2 studies)**

* [`AIM2_STUDY_PLAN.md`](AIM2_STUDY_PLAN.md): Aim 2 Study: Design Document
* [`AIM2_STUDY_SCORECARD.md`](AIM2_STUDY_SCORECARD.md): Aim 2 study scorecard, three seeds
* [`WORDLEVEL_STAGE0_REPORT.md`](WORDLEVEL_STAGE0_REPORT.md): Word-Level Stage 0: Alignment Reliability
* [`WORDLEVEL_STAGE1_SCORECARD.md`](WORDLEVEL_STAGE1_SCORECARD.md): Word-Level LFS, Stage 1: Scorecard
* [`CODESWITCH_SCORECARD.md`](CODESWITCH_SCORECARD.md): Code-switch study scorecard, three seeds
* [`AIM2_W0_RESULTS_2026-09-07.md`](AIM2_W0_RESULTS_2026-09-07.md): Aim 2 W0: offline offset and scale removal on the saved dumps. Results, 2026-09-07

**Decisions, audits, and corrections**

* [`SUBMISSION_DECISION_LFS_VS_RMFS.md`](SUBMISSION_DECISION_LFS_VS_RMFS.md): Submission decision: LFS versus RMFS
* [`ENGINEERING_DECISIONS_2026-08-17.md`](ENGINEERING_DECISIONS_2026-08-17.md): Engineering decisions (locked 2026-08-17)
* [`LITERATURE_NOVELTY_TABLE_2026-09-05.md`](LITERATURE_NOVELTY_TABLE_2026-09-05.md): Literature and novelty table for the LFS paper
* [`CONSISTENCY_AUDIT_2026-09-06.md`](CONSISTENCY_AUDIT_2026-09-06.md): Consistency audit, 2026-09-06
* [`DISCREPANCY_LEDGER.md`](DISCREPANCY_LEDGER.md): Discrepancy Ledger
* [`REVIEW_RESPONSE_2026-09-10.md`](REVIEW_RESPONSE_2026-09-10.md): Response to the 10 September 2026 review of the ICLR draft

## Study guide

* [`LFS_STUDY_GUIDE_2026-09-04.md`](LFS_STUDY_GUIDE_2026-09-04.md): LFS study guide: what to know, in what order, and how to defend it
* [`study/`](study/): the long-form study guide (`main.tex`, built as `main.pdf`, nine chapters from the
  machine-learning background to the numbers), `how_to_read_every_graph.pdf`, `LFS_FROM_THE_PROPOSAL.pdf`,
  and the figure scripts under `study/figs/`.
