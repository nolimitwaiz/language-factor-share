# rmfs: the stress-test program

The second phase of the project. It asked whether a single composite score (RMFS v1, a soft-minimum
over four per-language components) could summarize multilingual representation quality better than
LFS alone. The pre-registration was frozen on 2026-08-02 and the program was scored through the
A9 grid expansion. The composite failed its pre-registered crash test and did not generalize from
four to fourteen models, so the paper reports LFS as the core measurement and this program as the
stress test that establishes the limits of any single score. The decision is recorded in
`../docs/SUBMISSION_DECISION_LFS_VS_RMFS.md`.

What the paper still uses from here:

* the non-negative REML variance-component estimator **LFS-VC** (`src/rmfs/components/variance.py`),
  used for the 19-model expansion set and the content-transfer analysis;
* the expanded benchmark panels (33 models, Belebele and INCLUDE) and the covariate-adjusted
  validity analysis (`scripts/expanded_validity.py`, `src/rmfs/validity/`);
* the profile and hub audits behind the paper's tables (`scripts/audit_lfs_headline_claims.py`);
* the shared-preprocessing implementations of MEXA and AaR (`src/rmfs/metrics/baselines.py`);
* the ledgers, which record every decision, error, and run.

## Layout

    RULES.md                the twelve working rules, compute policy, coding standard
    pyproject.toml          package metadata; `pip install -e rmfs`
    Makefile                test and lint targets
    configs/
      models.yaml           registry generated from stored results: HF id, revision, layers, precision, dip layer
      datasets.yaml         NTREX with mandatory document ids; benchmark sources
      experiments/          one YAML per experiment
      new_models.tsv        the expansion-set candidates and the ones that ran
    prereg/
      PREREG_RMFS.md        the frozen pre-registration
      signatures/           numeric signature matrices
      addenda/              A2 to A9, dated, written before the affected runs
    src/rmfs/
      data/ntrex.py         loader that fails without document ids; purged splits
      components/           variance.py (REML components, LFS-VC), gates.py (content-variance
                            preservation, effective rank, norm), ladder.py (M0 to M5 map sequence,
                            one-standard-error selection), transfer.py (behavioral transfer),
                            shrinkage.py (Ledoit-Wolf)
      metrics/              rmfs.py (soft-minimum aggregation), baselines.py (MEXA and AaR under
                            shared preprocessing, plus the official-pooling MEXA column)
      validity/             deflation.py (covariate-adjustment regression), tournament.py
                            (model confidence set, orthogonalization, Romano-Wolf)
      battery/              synthetic fault injection and signature scoring
      utils/seeding.py      the single source of randomness
    scripts/                command-line entry points (see below)
      audit/                split-leakage and purged-decomposition audits
      sbatch/               the cluster job scripts, kept for provenance
    tests/                  unit tests for every component (`pytest rmfs/tests`)
    legacy/                 byte-identical, read-only copies of the prior pipeline (see legacy/README.md)
    ledgers/                DECISIONS.md, ERROR_LEDGER.md, INVENTORY.md (sha256 per legacy file), RUNS.md
    results/
      runs/<run_id>/        320 run directories, each with a manifest and its result.json
      tables/               the summary tables read by the paper
      figures/              figure PDFs
      belebele_new/, include_native/, belebele_arms/, deflation/, submission_comparison/,
      rmfs_v1_robustness/   benchmark scores and validity outputs
    paper/                  FINDINGS.md, appendix/estimator_notes.md, and the internal report sources

## Entry points

| Script | Purpose | Writes |
|---|---|---|
| `scripts/audit_lfs_headline_claims.py` | profile summary, estimator parity, hub audit | `results/tables/lfs_*_audit.*` |
| `scripts/layer_profiles.py`, `scripts/new_model_readings.py` | LFS-VC depth profiles for the expansion set | `results/runs/` |
| `scripts/grid_components.py` | variance components on a stored grid | `results/runs/` |
| `scripts/expanded_validity.py` | 33-model benchmark panel with covariate adjustment | `results/tables/expanded_validity.txt` |
| `scripts/compare_lfs_rmfs_submission.py` | same-support comparison of LFS and RMFS v1 | `results/submission_comparison/` |
| `scripts/translationese.py` | native-versus-translated check on WMT19 pairs | `results/runs/` |
| `scripts/rmfs_v1_robustness_audit.py` | robustness of the composite across its temperature | `results/rmfs_v1_robustness/` |

## Install and test

    pip install -e rmfs
    pytest rmfs/tests

The tests run on synthetic arrays and need no model or GPU. The rules the code follows are in
`RULES.md`; the project-wide research rules are in `../docs/00_research_rules.md`.
