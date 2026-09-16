# Aim 1 addendum: E-A dip-depth bootstrap, E-B FLORES-200 replication, E-C instruct versus base

**Frozen:** 2026-09-05, before any of the three runs was submitted. Approved by
Waiz Khan the same day ("run whatever experiments you need to run").
**Scope:** Aim 1 only. Forward passes and CPU analysis. No training.
**Estimator:** direct sums-of-squares LFS as in `code/pilot_metrics.py:lfs`,
mean pooling in fp32, joint per-coordinate z-scoring, exactly the
`code/cluster_grid.py` pipeline. Nothing in the estimator changes.
**Outputs never overwrite existing artifacts.** New roots:
`results/dip_bootstrap/`, `results/flores_grid/`, `results/instruct_grid/`.
Saved dumps under `results/dumps/` are read only.

## E-A. Bootstrap intervals on dip depth (CPU)

**Input.** The 14 campaign dumps (`results/dumps/<model>/layer000.npz` and
`layer<dip>.npz`, arrays `X` of shape 128 x 1500 x D, fp16 storage after
fp32 pooling; `meta.json` gives `dip_layer`).
**Procedure.** For each model: (1) parity canary: take the first 300
sentences, z-score jointly, compute LFS at layer 0 and at the dip layer, and
compare with `results/grid/<model>/metrics.json`; (2) draw 2,000 subsamples of
300 sentences without replacement from the 1,500 (the same 2,000 index sets
for every model, seed 20260905), compute LFS at layer 0 and at the dip layer
on each, and record dip depth = LFS(layer 0) minus LFS(dip); (3) per-language
contribution to the language sum of squares at the dip layer on all 1,500
sentences (E-D), `N * ||mu_l - mu||^2 / SS_lang`.
**Assembly.** Per model: median and 2.5/97.5 percentiles of dip depth and of
the two LFS values. Across models: Spearman between the point-estimate
ordering of dip depth (grid `metrics.json`) and each replicate's ordering.
**Predictions.**
- P-A1 (parity): |LFS(first 300 from dump) - LFS(grid)| < 0.02 at both layers
  for all 14 models. Failure means the dumps are not the grid's data and E-A
  is not reportable against the headline numbers.
- P-A2 (precision): the 95 percent interval half-width of dip depth is below
  0.03 for every model.
- P-A3 (ordering): Spearman between the point-estimate ordering and the
  replicate ordering is at least 0.90 in at least 95 percent of replicates.
**Reporting.** Intervals go into Table 1 and Figure 1b of the paper as an
uncertainty column whatever their width.

## E-B. FLORES-200 replication of the 17-model profile (GPU)

**Input.** FLORES-200 devtest (`flores200_dataset.tar.gz` from the public
NLLB release), first 300 sentences, every NTREX-128 language that maps to a
FLORES-200 code by an explicit table plus unambiguous ISO 639-3 prefix match.
Regional English and French variants in NTREX (eng-GB, eng-IN, eng-US, fra-CA)
have no FLORES counterpart and are excluded; the mapping table is written to
`data/FLORES200/ntrex_to_flores.json` before the run.
**Models.** The 17 primary-cohort models at their cached revisions.
**Procedure.** `code/cluster_grid_corpus.py` (a copy of `cluster_grid.py`
with a data-directory argument, an output-root argument, and a manifest
writer; estimator code imported unchanged), all layers, bf16 on A100 (fp32
for bloom-1b7, matching the original grid), batch size 32 for models up to
4B and 16 above. Smoke test on Qwen3-0.6B-Base with 8 languages and 50
sentences must finish and write finite metrics before the array runs.
**Predictions.**
- P-B1: at least 110 of the 128 NTREX languages map to FLORES-200.
- P-B2: endpoint recovery (interior minimum, both endpoints higher) in 17 of
  17 models on FLORES.
- P-B3: Spearman between NTREX and FLORES dip depth across the 17 models is
  at least 0.80.
- P-B4: the minimum layer agrees within two layers for at least 14 of 17.
- P-B5: median absolute difference in dip depth between corpora is below 0.05.
**Reporting.** Cross-corpus agreement enters the paper as one sentence and
an appendix table regardless of direction. If P-B2 or P-B3 fail, the
limitation is stated in the main text.

## E-C. Instruction-tuned versus base (GPU)

**Pairs.** Qwen3-0.6B, 1.7B, 4B, 8B (Base versus the post-trained release
of the same name), Qwen2.5-7B versus Qwen2.5-7B-Instruct, Llama-3.1-8B
versus Llama-3.1-8B-Instruct. Six pairs; Qwen2.5-7B base is run on the same
grid because the expansion cohort measured it under a different protocol.
**Procedure.** Same as E-B on NTREX-128, 300 sentences, all 128 languages.
Downloads are recorded in the manifest with snapshot hashes.
**Predictions.**
- P-C1: endpoint recovery holds for all six instruct models.
- P-C2: final-layer LFS is higher in the instruct model than in its base for
  at least four of six pairs (the output becomes more language-specific).
- P-C3: the minimum layer agrees within two layers for at least four of six.
- P-C4: |dip depth(instruct) - dip depth(base)| < 0.05 for at least four of six.
**Reporting.** One paragraph and an appendix table. Either direction is a
finding.

## Stop rules

A job that fails a finite-value check, a shape check, or the parity canary is
reported as failed and not rerun with changed settings. No model is replaced
after outcomes. Nothing here changes a frozen criterion of any earlier study.

## Execution record

- 2026-09-05: files pushed to CLSP (`code/cluster_grid_corpus.py`,
  `code/prepare_flores.py`, `src/analysis/dip_bootstrap.py`, four sbatch
  scripts). `dip_bootstrap.py` passed a synthetic parity test locally
  (exact match, CPU, fake dump).
- FLORES-200 devtest prepared on CLSP by `prepare_flores.py`: 1,012 sentences,
  **115 of 127 NTREX codes mapped** (P-B1 PASS). Unmapped: div, eng-GB,
  eng-IN, eng-US, fra-CA, hmn, mey, nde, shi, tah, ton, ven. Mapping frozen in
  `data/FLORES200/ntrex_to_flores.json`.
- Submitted 2026-09-05 (EDT, evening): E-A array **1784646** (cpu, 14 tasks);
  E-B smoke **1784647** (gpu-a100, Qwen3-0.6B-Base, 8 languages, 50
  sentences); E-B array **1784648** (gpu-a100, 17 tasks, `afterok:1784647`);
  E-C array **1784649** (gpu-a100, 7 tasks).
- **All 39 tasks COMPLETED with exit 0:0** (2026-09-05, EDT morning). E-A
  tasks ran 0:46 to 4:15 on cpu nodes c02/c04/c10/c11/c24/c25; E-B tasks 7:40
  to 51:47 and E-C tasks 13:42 to 28:39 on A100 node e01. No task was rerun
  and no model was replaced.
- Scored by `src/analysis/score_eabc.py` on CLSP; `results/eabc_scoring/summary.json`
  SHA-256 `fcdf3d6f707bfe09ccd392f95486dba30bf7fce4baef369057bb0239ad7a3d14`.
- **Verdicts:** P-A1 PASS (max parity gap 7.8e-5), P-A2 **FAIL** (12/14; Mistral-7B
  half-width 0.045, Salamandra-2B 0.059), P-A3 PASS (100% of replicates);
  P-B1..P-B5 all PASS (115 languages; 17/17; Spearman 0.917; 14/17 within two
  layers; median |depth difference| 0.0105); P-C1..P-C4 all PASS (6/6; 5/6;
  5/6; 6/6). Eleven of twelve. Full write-up:
  `docs/AIM1_EABC_RESULTS_2026-09-05.md`.
