# RMFS — Robust Multilingual Factor Score

> **Consolidation note (2026-08-16):** This campaign is now stored under the
> canonical `multilingual-metrics/rmfs/` directory. Its frozen protocols,
> code, results, and ledgers are preserved. Active submission framing lives in
> the project-root `CLAUDE.md`, `MEMORY.md`, `CODEX.md`, and
> `docs/SUBMISSION_DECISION_LFS_VS_RMFS.md`. The current decision is to submit
> LFS as the core contribution and report RMFS as a stress-test campaign, not
> as a superior scalar.

Successor-metric research for the Koehn–Murray NSF proposal on LLM
multilinguality. Start with `CLAUDE.md` (rules; they override convenience),
then `MEMORY.md` (history, numbers, artifact map), then
`CLAUDE_CODE_PROMPT.md` (the phased build plan).

**Status: prereg FROZEN (`prereg-rmfs-v1`, 2026-08-02). The full campaign was
scored through the A9 grid expansion. The preregistered scalar failed its
crash test and did not generalize from four to fourteen models; the individual
readings and the validation protocol remain the scientific output.**

## Repository map — and where old code ends and new code begins

**Old code lives in `legacy/` and only there** — byte-identical, read-only
copies of the prior LFS-campaign pipeline (chmod a-w, sha256 per file in
`ledgers/INVENTORY.md`). It is never edited; it is wrapped, ported, or
cited. **New code lives in `src/rmfs/`** and is held to exact numerical
parity with legacy where it ports it (`tests/test_ladder.py`).

    CLAUDE.md               rules (12 non-negotiables) + proposal objectives
    CLAUDE_CODE_PROMPT.md   the phased build plan (P0-P4)
    MEMORY.md               project memory: history, campaign numbers, map
    pyproject.toml          py>=3.11, ruff, pytest; cluster extra for torch
    Makefile                reproduce-tables stub (populated in Phase 4)

    prereg/                 FROZEN pre-registration + signatures (75 cells)
      addenda/              dated addenda only (A1 = capacity gates, pending)
    configs/                models.yaml (17 models, GENERATED from committed
                            results), datasets.yaml (doc IDs mandatory)

    src/rmfs/               NEW code, typed, tests-first
      data/ntrex.py           loader (hard-fails w/o doc IDs) + purged splits
      components/variance.py  REML components, LFS-VC, LDE, legacy floor
      components/gates.py     C_pres, effective rank, norm, dPPL hook
      components/shrinkage.py Ledoit-Wolf (typed wrapper)
      components/ladder.py    M0-M5 port + GPA, purged cross-fit, one-SE K
      metrics/  validity/  battery/  extraction/   (Phases 1.6-3)

    tests/                  live unit battery (44 green): closed forms,
                            rung recovery, EXACT legacy parity, hard-fails
    legacy/                 OLD code, read-only (see legacy/README.md)
      prereg/               the six frozen prior-campaign preregs
      tests/                prior tests, reference only

    scripts/
      audit/                Phase-0 split-leakage audits (E1 evidence)
      sbatch/               cluster job scripts (Claude submits them)
      scaffold.sh           one-time repo bootstrap, kept for provenance
    ledgers/                INVENTORY, ERROR_LEDGER (E1-E2), RUNS, DECISIONS
    paper/appendix/         estimator_notes.md (N1-N4), grows as built
    data/                   gitignored; external/ covariates + Belebele,
                            embeddings/ is a POINTER to cluster dumps (65G)
    results/                gitignored except runs/<id>/manifest.json

Cluster mirror: `wkhan12@login.clsp.jhu.edu:~/rmfs` (rsync; no model
compute happens locally, ever).
