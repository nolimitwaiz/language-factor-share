# ERROR_LEDGER

## E1 (2026-08-01) — Misalignment-budget splits were never document-purged; leakage is near-total

**Phase 0.3a. Affects a published table (the misalignment budget, technical
report §"Construct Boundary", 13 model primaries).**

**What the legacy code does.** `legacy/budget_real.py` cross-fits the M0–M5
ladder over S=20 splits built by `stratified_split`: English source
sentence-length terciles, `rng.permutation` within each tercile, half to
train. Document identity is never consulted anywhere in the budget path;
`DOCUMENT_IDS.tsv` is read only by the §3.1.3 transfer scripts.

**Quantified leakage** (`scripts/audit_leakage.py`, verbatim replication of
the split generator — same seed, same rng-consumption order — joined with
NTREX document IDs; 1997 sentences in 123 documents, ~16 sentences each):

| protocol | docs | documents straddling train/held-out | held-out sentences sharing a document with train |
|---|---|---|---|
| n=300 (dev) | 20 | mean 97.0% [90.0–100%] | mean **99.3%** [95.4–100%] |
| n=1500 (campaign) | 91 | mean 99.5% [98.9–100%] | mean **99.9%** [99.2–100%] |

Effectively every held-out sentence shares a document — hence topic,
vocabulary, entities, and discourse context — with the training side. The
"held-out" reconstruction losses that select ladder rungs and produce the
published shares were computed on near-in-sample text.

**Why it matters, and why it may not.** Leakage inflates held-out fit
quality for every rung simultaneously. The published *shares* are
between-rung differences, so a uniform inflation could largely cancel; the
audit therefore does not assume damage in either direction. Whether the
offset/scale/rotation/linear shares actually move is an empirical question
answered by the 0.3b re-estimate (Qwen3-1.7B, n=1500, published rank
reused, splits changed and nothing else; decision rule fixed in advance:
any rung moving >2 percentage points = the published table changes).
Result: appended below when job 1703653 lands.

**Root cause class.** Same as prior-campaign D4/D6/D8/D10: a procedure
(random sentence splits) carried into a setting (document-structured
corpus) without re-checking its premise. NTREX adjacency was known — the
§3.1.3 harness explicitly exploits same-document context — and the split
code never consulted it.

**Standing fix.** RULES.md rule 4 (purged splits everywhere); every new
loader carries document IDs; `purged_split` (documents stratified by median
length-stratum, sentences follow their document) is the Phase-1 default.

## E2 (2026-08-01) — DOCUMENT_IDS.tsv was never on the cluster; the legacy §3.1.3 loader degrades silently without it

**Found** when audit job 1703711 died on FileNotFoundError: the cluster copy
of the prior repo has no `data/NTREX/DOCUMENT_IDS.tsv`. The file existed
only locally.

**Why this matters beyond the audit.** `legacy/transfer_313b.py:load_pairs`
guards the file with `if os.path.exists(docp)` and, when absent, silently
builds sentence pairs WITHOUT the same-document constraint. Every
cluster-side run of the 3.1.3 family therefore ran unfiltered: the
transfer/attention harness runs and, in the successor campaign, every
`context_use*` evaluation over the training arms. Two consequences,
bounded but real:

1. Pair selection differed from the documented design (the first N
   qualifying pairs are taken; without the filter, pairs that straddle the
   ~122 document boundaries in 1997 sentences are eligible — order a few
   percent of candidates). Matched contexts are adjacent sentences either
   way; a boundary-straddling "matched" context weakens that condition
   slightly, biasing measured context gains DOWN, not up.
2. The code comment's claim that mismatched partners are "different
   document by construction" held only where the file existed.

**What is unaffected.** All arm-vs-control contrasts in the prior campaign
used identical pair sets within each run, so within-run comparisons stand.
ε_B in the RMFS prereg derives from percentiles of the v2 distributions and
is robust to marginal pair-set differences; its provenance note stands.

**Fixes.** (a) The TSV is now on the cluster (dataset copy completed, raw
data added, nothing modified). (b) New-stack rule, adopted now: RMFS
loaders HARD-FAIL when document IDs are missing — the silent-degrade guard
(`if os.path.exists`) is the direct cause here and is banned in new code
(RULES.md rule 4 enforcement detail). (c) Prior-campaign impact is noted
here rather than re-litigated; if the 3.1.3 harness is re-run in Phase 2,
it runs with the filter and the delta gets measured, not assumed.

**Root cause class.** Silent degradation on missing input — the same family
as the prior campaign's D10 (silent skip on a name mismatch). A guard that
turns missing data into a quiet behavioral change produces plausible output
with no error, which is why it survived a full campaign.

### E1 resolution — purged re-estimate (job 1703744, Qwen3-1.7B, n=1500, published rank r=64 reused)

**Parity first: the replication is exact.** The legacy-splits arm reproduces
the published grid-mean shares to three decimals on every rung (37.212 /
28.611 / 0.107 / 4.218 / −10.750), so the wrapper is generator-for-generator
faithful and the purged deltas below are attributable to the splits alone.

| rung | published | purged | Δ (pp) | >2pp? |
|---|---|---|---|---|
| M1 offset | 37.212 | 36.079 | −1.133 | no |
| M2 scale | 28.611 | 27.794 | −0.817 | no |
| M3 rotation | 0.107 | 0.079 | −0.028 | no |
| M4 linear | 4.218 | 4.820 | +0.601 | no |
| M5 nonlinear | −10.750 | −16.771 | **−6.020** | **FLAG** |

**Verdict.** The substantive published conclusions SURVIVE purging: real
misalignment remains overwhelmingly additive (M1+M2 ≈ 64%, moving ~1pp),
rotation remains negligible (0.08–0.11%), linear remains small (4–5%).
The flag fires on M5 only: the nonlinear rung's held-out share, already
negative in the published table (the kernel map overfits), is **6pp more
negative** under document-purged evaluation. The mechanism is exactly the
leakage: the most flexible map benefits most from near-in-sample held-out
text, so leaky splits understated its overfitting. A published number
changes; the published narrative (M5 treated as overfit) does not reverse —
it strengthens.

**Consequences.** (1) The prior repo's report should carry a correction note
for the M5 row (queued as a prior-campaign erratum, not an RMFS task).
(2) For RMFS this validates the prereg's design choice directly: one-SE rung
selection on PURGED held-out loss will select M5 even less often, which is
the conservative direction. (3) The Phase-1 ladder-parity tolerance
(1.0pp/rung on unpurged replication) is comfortably achievable — parity here
was exact.

## E4 — frozen signature-table defects exposed by the S2 battery (2026-08-04)
Three cells in prereg/signatures/rmfs_signatures.csv encode expectations
that are wrong or self-contradictory as written (offset q_K rung_le M1
unsatisfiable vs base M2; collapse_per_language q_L direction contradicts
its own note; pairing C_pres band assumes a per-language estimator while
the declared estimator is pooled REML). All three scored as frozen and
FAILED; none reinterpreted post hoc. Impact: 3 of the 26 failed cells are
table defects, not instrument responses. The table is frozen — corrections
may only appear as a dated addendum BEFORE any future battery run.
