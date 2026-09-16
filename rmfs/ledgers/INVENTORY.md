# INVENTORY

Phase 0.1 artifact inventory. Compiled 2026-08-01 from a read-only sweep of
the local machine and the CLSP cluster. Raw artifacts were not moved or
renamed; `legacy/` holds byte-identical copies (one renamed, noted below).

## Corrections to the kickoff pack (verified, they change execution)

1. **Six legacy paths in the kickoff are stale.** `budget_real.py`,
   `ladder.py`, `inject.py`, `gpa.py`, `subspace.py`, `teeth.py` live under
   `multilingual-metrics/src/{analysis,geometry,battery}/`, not `code/`.
2. **`study_metrics.py` never existed** in the prior repo; the file is
   `code/pilot_metrics.py` (the pilot→study rename touched prose, not
   filenames, because the module is imported everywhere). Copied
   byte-identical here as `legacy/study_metrics.py`, the name RULES.md and
   the kickoff reference.
3. **"28 Aim-2 arms" = 25 trained + 3 frozen references.** The cluster holds
   75 checkpoints (25 arms × 3 seeds, 168G). A, CS-A and WA-A are the frozen
   base model, evaluated from base weights; they have result JSONs but no
   checkpoint directories. Nothing is missing.
4. **No embedding dumps exist for any training arm** — checkpoints only.
   The kickoff's claim that arm dumps exist is false. Phase 2.2 (T on all
   arms) therefore requires extraction jobs from checkpoints; budget those
   before Phase 2.
5. **Cluster shared volume is ~97% full** (2.4T free on the shared mount)
   even though personal quota shows ~440G headroom (559G used / 1000G soft).
   Any new dump campaign needs a size estimate against the shared volume,
   not just quota.
6. **The misalignment-budget splits are not document-purged** (see
   ERROR_LEDGER, Phase 0.3 audit). Stratifier is English sentence-length
   terciles; `DOCUMENT_IDS.tsv` is read only by `transfer_313{,b}.py`.
7. Local `results/dumps/` in the prior repo is empty (a single log file);
   every heavy artifact is cluster-side only.

## Legacy code (copied into `legacy/`, chmod a-w, byte-identical)

sha256 prefixes; provenance = `~/Desktop/multilingual-metrics/` + path.

| file | sha256[:16] | provenance | imports standalone? |
|---|---|---|---|
| study_metrics.py | 3c1b2fa743a7b6e3 | code/pilot_metrics.py (renamed copy) | yes (numpy/torch only) |
| cluster_grid.py | 20ae81c522eae860 | code/cluster_grid.py | yes (data loader) |
| attribution.py | 66116cbde2bda37b | code/attribution.py | needs result CSVs |
| lens_312.py | 5890956a37963e42 | code/lens_312.py | needs model + NTREX |
| transfer_313b.py | a637b8f4f0aaacce | code/transfer_313b.py | needs model + NTREX |
| attention_313.py | eedf343520694cde | code/attention_313.py | imports src.common.ledger — reference-only |
| fertility.py | 98dd770948126b2e | code/fertility.py | needs tokenizers |
| budget_real.py | ae3722663ad9ef66 | src/analysis/budget_real.py | imports src.common.ledger — reference-only |
| teeth.py | 1348d07b1192b418 | src/analysis/teeth.py | reference-only |
| ladder.py | 9b20782c4abe53d3 | src/geometry/ladder.py | package-relative imports — load via package shim |
| gpa.py | 9804317225e54bd6 | src/geometry/gpa.py | package-relative imports — load via package shim |
| subspace.py | 129bbd8cae5cd83b | src/geometry/subspace.py | package-relative imports — load via package shim |
| inject.py | e718765e2638824f | src/battery/inject.py | package-relative imports — load via package shim |
| dump_embeddings.py | c44a209e149d352f | src/campaign/dump_embeddings.py | reference-only (dump format spec) |

"reference-only" = imports `src.common.ledger` or other prior-repo modules;
port rather than import in Phase 1.

Correction (2026-08-01, found when the first 0.3b submission died on
ImportError): `ladder.py`, `gpa.py`, `subspace.py`, `inject.py` use
package-relative imports (`from .gpa import ...`) and are NOT standalone as
first recorded; they load only inside a package context. The audit script
synthesizes a `legacypkg` package rather than editing read-only legacy
files. Failed run + resubmission logged in RUNS.md.

`legacy/tests/` (8): conftest.py, test_estimator_floor.py, test_gpa.py,
test_inject.py, test_ladder.py, test_parity.py, test_pooling.py,
test_subspace.py — from `tests/`.

`legacy/prereg/` (6): PREREG_BATTERY.md, PREREG_ATTENTION_313.md,
PREREG_WORDLEVEL.md, PREREG_CODESWITCH.md, PREREG_WORDALIGN.md (from
`prereg/`), PREREG_312_313.md (from `paper/`). These are the frozen prior
campaign preregs; frozen text retained verbatim including its known-wrong
statements (D8) with ledger entries attached in the prior repo.

## Saved embeddings (cluster-side ONLY — do not copy locally)

`wkhan12@login.clsp.jhu.edu:~/multilingual-metrics/results/dumps/` — 65G,
14 model directories, n=1500 sentences, 5 sampled layers + per-sentence
margins per model, fp16 npz with fp32-pooled statistics:

bloom-1b7, bloom-7b1, EuroLLM-1.7B, Falcon3-7B-Base, Mistral-7B-v0.3,
OLMo-2-0425-1B, OLMo-2-1124-7B, Qwen3-{0.6B,1.7B,4B,8B}-Base,
salamandra-2b, salamandra-7b, SmolLM2-1.7B.

Format per model: `layerNNN.npz` ×5 + `marginsNNN.npz` ×5 + meta (layer
indices differ per model, e.g. Qwen3-0.6B: 000/007/008/021/028). Reader
spec: `legacy/dump_embeddings.py`. Per-model size 1.8G→6.9G.
Missing from dumps vs the 17-model grid: granite-3.1-8b, Yi-1.5-9B,
Llama-3.1-8B (grid metrics exist as JSONs; raw dumps were not saved for
these three).

`data/embeddings/README.md` records this pointer; the directory holds no
data by design.

## Aim-2 arm checkpoints (cluster-side)

`~/multilingual-metrics/results/aim2/ckpt_*` — 75 directories, 168G, HF
format (model.safetensors + tokenizer), Qwen3-0.6B-Base fine-tunes,
seeds {0,1,2}:

- LFS study: armB, armC, armD, armE, armF (arm A = frozen, no ckpt)
- Guard sweep + tail: armE_cvp10, armE_cvp100, armE_cvp1k, armG_tail
- Code-switch 400-step: armCS-{B,W10,W25,W50,P25} (CS-A frozen)
- Code-switch 2000-step: armCS-{B,W10,W25,W50,P25}_s2000
- Word-align: armWA-{B,C,D,E,F,G} (WA-A frozen)

Note: arm seeds are {0,1,2}, NOT the new-repo default {13,42,71}; any
re-analysis of the arms keeps their original seeds.

Alongside: per-arm training logs (`Qwen3-0.6B-Base_arm*_seed*.json`) and
downstream evaluations (`evaluation*.json`, `context_use*.json` for
base/sweep/codeswitch/codeswitch_frozen/wordalign). Known gap: the
2000-step code-switch arms have checkpoints but NO downstream evaluation.

## Downstream + covariate tables (local, copied where needed)

- Belebele: `multilingual-metrics/results/belebele0/` — 19 models incl.
  aim2-control/aim2-lfs; `results/belebele/` — 9-model subset.
- Deflation covariates → copied into `data/external/` (7 files):
  fertility.csv, tokens_proxy_culturax.csv, cc100_sizes.csv,
  language_meta.csv, language_meta_backstop.csv, mexa_flores_paper.csv,
  mexa_flores_max_9models.csv.
- §3.1.3 transfer results: `multilingual-metrics/results/transfer313/` —
  4 models × {v1, v2} JSONs (source for the prereg ε_B fill).
- 17-model grid metric JSONs: `multilingual-metrics/results/grid/`.
- Misalignment budget: `multilingual-metrics/results/budget/` — 41 variants
  (13 model primaries + fixed-rank + centered variants), the published
  shares the Phase 0.3 audit compares against.

## Datasets

- NTREX-128: `multilingual-metrics/data/NTREX/NTREX-128/` (128 language
  files) + `data/NTREX/DOCUMENT_IDS.tsv` (1997 lines, **123 unique
  documents**, ~16 sentences/document). Document IDs are mandatory in every
  new loader (RULES.md rule 4).
- Code-switch corpora (Aim-2 §3.2.1): cluster
  `~/multilingual-metrics/data/codeswitch/` — 6 configurations + manifest
  with per-language aligner agreement.
- Word alignments (20 languages, two-aligner agreement recorded):
  `multilingual-metrics/results/wordlevel/alignments/` (local + cluster).
