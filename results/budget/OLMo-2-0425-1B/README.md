# OLMo-2-0425-1B: no campaign-scale selected-rank budget

The campaign (n=1500) selected-rank primary for this model did not
complete. Its permutation-null computation exceeded a 60-hour allocation
twice (SLURM 1691973_12) and was not rerun.

The file in this directory is the **n=300 prototype result**, renamed to
`budget.STALE_n300.json` so that scripts globbing `budget.json` fail
loudly rather than silently mixing scales. It is identifiable by
`selected_rank: 16`, a rank that does not occur at campaign scale.

Complete campaign budgets for this model exist in the sibling
directories `OLMo-2-0425-1B_fixed_raw/` (common rank 64) and
`OLMo-2-0425-1B_selected_centered/`. Both place its rotation share below
zero. The technical report states this exclusion explicitly.
