# Research rules the project followed

These rules were fixed at the start of the project and applied to every run and every document.
They explain several choices a reader will notice in the code and the results.

## Non-negotiable rules

1. Frozen pre-registrations and their signatures are immutable. Changes go only in a dated
   addendum written before the affected run.
2. No result is fabricated, interpolated, silently omitted, or relabeled.
3. Confirmatory and exploratory findings are kept visibly separate.
4. Raw artifacts and the producing code are consulted before summaries or PDFs.
5. Cross-validation over NTREX sentences uses document-purged splits, because adjacent sentences
   share context.
6. Pooling and statistics are computed in fp32. BLOOM is never run in fp16.
7. Saved embeddings and checkpoints are never overwritten; new extractions get new run ids.
8. Every run writes a manifest with code hash, configuration, input hashes, seed, host, precision,
   and wall time.
9. LFS is reported with its language, concept, and residual components, grid size, pooling,
   preprocessing, layer, model revision, and an uncertainty statement.
10. No causal claim about downstream improvement is drawn from an observational correlation.
11. Lower LFS is never called universally better.
12. Representational collapse is not called harmful unless downstream harm is measured. The 0.6B
    Belebele test found large reorganization without measured harm.
13. No "first of its kind" claim is made without a formal literature review
    (`LITERATURE_NOVELTY_TABLE_2026-09-05.md`).

## Compute rules

* The local machine is used for editing, documentation, and CPU checks only. Model extraction,
  training, inference, and table rebuilds run on the cluster.
* Existing dumps are read-only.
* Any cluster job above about one GPU-hour, any model download above 10B parameters, any
  pre-registration freeze, and any change to a frozen criterion require explicit approval from
  the project lead.

## Measure boundaries

* **LFS** is the direct sums-of-squares estimator, `SS_lang / (SS_lang + SS_con)` after joint
  per-coordinate standardization on a balanced language-by-sentence grid. Lower LFS means that
  sentence variation dominates language variation in this additive decomposition; it does not by
  itself mean a more capable model.
* **LFS-VC** is the REML variance-component estimator. Its favorable-direction form is
  `1 - LFS-VC`. The two estimators are never called identical: their rankings agree closely on
  the same support, but the levels differ.
* **MEXA** (Kargaran et al., 2024) is a published reference measure, reimplemented under the same
  preprocessing. It is reported for reference and is never presented as an original component.
* **RMFS v1**, the pre-registered composite of the stress-test program, failed its crash test and
  is reported as a scored negative result, not as a validated score (`rmfs/README.md`).
