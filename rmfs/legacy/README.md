# legacy/ — the PRIOR pipeline. READ-ONLY. Old code lives here and ONLY here.

Byte-identical copies from `~/Desktop/multilingual-metrics` (the LFS
campaign repo), frozen at copy time; per-file sha256 + provenance:
`ledgers/INVENTORY.md`. One rename: `study_metrics.py` is
`code/pilot_metrics.py` there (the name the kickoff references).

Rules (CLAUDE.md rule 7):
- NEVER edit these files. Fixes, typing, and improvements happen in ports
  under `src/rmfs/`; discrepancies get ledger entries first.
- `gpa.py`, `ladder.py`, `subspace.py`, `inject.py` use package-relative
  imports and do NOT import standalone — load them through the package
  shim (see `scripts/audit/audit_purged_budget.py`) or import their PORTED
  equivalents from `src/rmfs/components/`.
- The port of ladder/GPA is held to EXACT numerical parity by
  `tests/test_ladder.py`; if you think legacy is wrong, the ledger entry
  comes before any change of behavior in the port.

`prereg/` here holds the SIX frozen pre-registrations of the prior
campaign, verbatim, including their known-wrong statements (documented in
the prior repo's discrepancy ledger). `tests/` holds the prior unit tests
for reference; the live battery is the top-level `tests/`.
