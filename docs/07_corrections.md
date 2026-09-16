# Corrections ledger, in brief

Every discrepancy found during the project is recorded with its date, cause, and fix in
`DISCREPANCY_LEDGER.md` (D1 to D14). Entries are appended, never edited away. In brief:

| Id | What was found | Fix |
|---|---|---|
| D1 | A torn line in the cluster run ledger | file locking on append |
| D2 | A stale prototype result left in a results directory | renamed so that globs fail loudly |
| D3 | A size-invariance claim stated too strongly | replaced by the exact expected value and its checks |
| D4 | A permutation-null criterion written with an inverted inequality | corrected before scoring |
| D5 | Function words contaminating the word-level concept grid | filtered at the concept definition |
| D6 | An Aim 2 criterion that the control condition also failed | criterion rewritten |
| D7 | A criterion scored on sign alone | replaced by an effect-size criterion |
| D8 | A preregistration that mis-described its own held-out set | frozen text left as written; correction applied in the analysis |
| D9 | An inert variance-preservation term in the word-level trainer | fixed to use a frozen reference and rerun at three seeds |
| D10 | A frozen reference skipped by name in an evaluation script | identified by set membership instead |
| D11 | Profile counts that combined the two estimators | the two model sets reported separately since |
| D12 | A hub result stated as universal when it is model-specific | reported as a range, 113 to 124 of 127 |
| D13 | A regression statistic quoted from the development-grid run | 17-model artifact values reported |
| D14 | An offset-plus-scale range quoted from a three-model prototype | 63 to 97 percent across the 14-model study |

The external review responses that led to the September corrections are in
`REVIEW_RESPONSE_2026-09-10.md`; the consistency audit of reported numbers against artifacts is in
`CONSISTENCY_AUDIT_2026-09-06.md`.
