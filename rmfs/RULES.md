# Rules for the rmfs package

These are the working rules of the stress-test program (the second phase of the project, which
tested whether a single composite score could summarize multilingual representation quality).
Code comments and ledger entries under `rmfs/` cite them by number.

## Non-negotiable rules

1. **The pre-registration is law.** `prereg/PREREG_RMFS.md` and everything under
   `prereg/signatures/` are frozen once signed. Frozen files are never edited. Changes go in dated
   addenda under `prereg/addenda/`, written before the affected run.
2. **No RMFS number is computed before the pre-registration is frozen.**
3. **Never fabricate, interpolate, or reconstruct a result.** If a run failed or a number is
   missing, the ledger says so. A blank cell beats an invented one.
4. **Purged splits everywhere.** Any cross-validation over NTREX sentences purges at the document
   level, because adjacent sentences share context. Concept-level holdouts are also
   document-disjoint. This is a correctness requirement, not a preference.
5. **Precision discipline.** All pooling and statistics in fp32 (fp16 mean-pooling overflows on
   massive activations). BLOOM inference in fp32 or bf16 only, never fp16. Both caused real
   incidents; they are in the error ledger.
6. **Shared preprocessing for measure comparisons.** Any head-to-head between measures uses
   identical embeddings and preprocessing. The official-pooling MEXA (position-weighted token
   averaging, per Kargaran et al.) is a separate, clearly labeled column and is never mixed into
   shared-preprocessing comparisons.
7. **Legacy is read-only.** `legacy/` holds byte-identical copies of the prior pipeline
   (`study_metrics.py` is `code/pilot_metrics.py` renamed; `cluster_grid.py`, `attribution.py`,
   `lens_312.py`, `transfer_313b.py`, `attention_313.py`, `fertility.py`, `budget_real.py`,
   `teeth.py`, `ladder.py`, `gpa.py`, `subspace.py`, `inject.py`, `dump_embeddings.py`, the prior
   tests, and the prior pre-registrations). It is imported, wrapped, or ported, never edited in
   place. Provenance and a sha256 per file: `ledgers/INVENTORY.md`.
8. **Saved embeddings are never overwritten.** New extractions get new run ids.
9. **Every run writes a manifest**: configuration snapshot, code hash, seed, host, precision,
   wall time, and input-artifact hashes, under `results/runs/<run_id>/`, plus one line in
   `ledgers/RUNS.md`.
10. **Failures are findings.** Failed predictions, gate failures, and bugs go verbatim into
    `ledgers/ERROR_LEDGER.md` with date and impact.
11. **Seeds.** Three seeds for anything stochastic, seed set {13, 42, 71} unless the
    pre-registration says otherwise. All randomness goes through `src/rmfs/utils/seeding.py`.
12. **Approval gates.** Freezing a pre-registration, launching any cluster job estimated above about
    one GPU-hour, downloading any model above 10B parameters, and changing any frozen criterion all
    require explicit approval from the project lead first.

## Compute policy

- **Local machine**: editing and CPU unit tests on synthetic arrays only. No model forward pass
  runs locally, not even smoke runs. Anything touching model weights or saved embeddings runs on
  the cluster, where the dumps live.
- **Cluster**: a SLURM cluster with RTX-class nodes (fp16 inference; fp32 for BLOOM-1.7B) and
  A100-80GB nodes (bf16). Routing policy since 2026-08-02: submit to the A100 partition first with
  short wall times (at most 3 hours) and modest footprints so that backfill scheduling picks the
  job up; the cpu and gpu partitions are fallbacks. Job scripts live under `scripts/sbatch/` for
  provenance.
- **Data**: NTREX-128 with document ids (1,997 sentences in 123 documents, about 16 per document;
  the ids are carried through every loader), Belebele (zero-shot log-likelihood), CulturaX token
  counts with CC-100 as a fallback exposure proxy, Glottolog-style macro-families, and per-model
  tokenizer fertility.
- **Model grid**: 17 evaluated models (Qwen3 0.6B, 1.7B, 4B, 8B; OLMo-2 1B, 7B; BLOOM 1.7B, 7B;
  Salamandra 2B, 7B; EuroLLM-1.7B; SmolLM2-1.7B; Falcon3-7B; Mistral-7B-v0.3; granite-3.1-8B;
  Yi-1.5-9B; Llama-3.1-8B). Hidden-state dumps: 14 models, 1,500 sentences, five layers plus
  margins, about 65 GB, cluster-side only. Training conditions: 28 named, of which 25 were trained
  (75 checkpoints, about 168 GB, cluster-side) and 3 are frozen references evaluated from base
  weights. The unseen model families named in the pre-registration are not touched by any analysis
  before the final phase.

## Coding standard

- Python 3.11 or later, typed, `ruff` clean. Pipeline logic lives in `src/`, never in notebooks.
- Every component function takes arrays plus a configuration and returns a dataclass with the value
  and its diagnostics (sample sizes, floors, flags). No bare floats. Negative variance components
  are counted and reported, never silently clipped.
