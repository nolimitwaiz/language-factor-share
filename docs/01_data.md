# Data

Nothing here was collected by us; every corpus and benchmark is public. The repository stores only
small metadata files. Download instructions and the exact subsets used are below.

## Parallel corpora (the grids)

**NTREX-128** (Federmann et al., 2022). The WMT19 English news test set translated by professional
translators into 128 languages, with document boundaries. License CC BY-SA 4.0.
Source: <https://github.com/MicrosoftTranslator/NTREX>. Place the checkout at `data/NTREX/` so that
`data/NTREX/NTREX-128/newstest2019-ref.<lang>.txt` exists; `code/cluster_grid.py` reads it there.

* Primary grids: the **first 300 sentences** of every one of the 128 references (`--n_sents 300`),
  giving a 128 x 300 grid per layer. Every primary-set profile, the hub test, and the instruction-tuned
  comparison use this grid.
* Extended dumps: the **first 1,500 sentences** for the 14 models marked in `docs/02_models.md`,
  written by `src/campaign/dump_embeddings.py` as one `.npz` per layer of interest. These feed the
  sentence-resampling intervals (`src/analysis/dip_bootstrap.py`: 2,000 subsets of 300 drawn without
  replacement from the 1,500, identical index sets for every model, at the stored layers), the
  misalignment decomposition (`src/analysis/budget_real.py`), the invariance suite, and the offline
  offset-removal pre-test. Document ids (`data/NTREX/DOCUMENT_IDS.tsv`) are used to make
  document-purged train and test splits wherever a fit is evaluated on held-out sentences.
* Training interventions use NTREX lines 500 to 1,899 (`TRAIN_RANGE` in `src/aim2/train.py`) in
  English, German, Hindi, and Swahili, disjoint from every evaluation subset.

**FLORES-200 devtest** (NLLB Team, 2022). Wikipedia-domain sentences; license CC BY-SA 4.0.
Source: <https://github.com/facebookresearch/flores>. Used for the second-corpus replication:
the first 300 devtest sentences in the 116 NTREX languages that FLORES covers (English plus 115
others), mapped by `data/FLORES200_ntrex_to_flores.json`; eight NTREX languages absent from FLORES
and the regional English and French variants are excluded. Preparation: `code/prepare_flores.py`;
extraction: `code/cluster_grid_corpus.py`.

**WMT19 newstest** native and translated pairs, seven languages, 300 pairs per direction, for the
translationese check (`rmfs/scripts/translationese.py`).

## Behavioral targets

**Belebele** (Bandarkar et al., 2024). Reading-comprehension multiple choice, 300 questions per
language. Scored with EleutherAI's lm-evaluation-harness by log-likelihood over the four choices.
`results/belebele/` is the 5-shot run on the development grid (`cluster/submit_belebele.sh`);
`results/belebele0/` is the zero-shot run (`cluster/submit_belebele0.sh`); the expanded panels used
by the pooled-benchmark analysis are under `rmfs/results/belebele_new/`.

**INCLUDE** (Romanou et al., 2024). Natively authored exams in 30 languages that overlap our grid,
same scoring; `rmfs/results/include_native/`.

**Content-transfer test** (our protocol, frozen 2026-08-24). 100 document-distinct NTREX pairs:
an English target sentence and the sentence that precedes it in the same news story, the context
translated into 41 languages. For each model the teacher-forced fp32 loss on the target tokens is
measured with the true context and with a length-matched context from a different story (a
deterministic minimum-cost derangement). Pair construction: `src/analysis/build_r1_content_pairs.py`;
frozen pairs: `results/round3/r1_content_pairs_2026-08-24/`; scoring: `src/analysis/r1_content_transfer.py`.

## Covariates for the adjustment

All in `results/deflation/`:

* `tokens_proxy_culturax.csv`: per-language token counts in CulturaX (Nguyen et al., 2023), the
  training-data exposure proxy (log10 tokens in the regression).
* `fertility.csv`: tokenizer fertility, tokens per word, per model and language.
* `language_meta.csv`: script and language family per language; macro-families follow Glottolog.
* `cc100_sizes.csv`: an alternative exposure proxy kept for comparison.

## Sizes at a glance

| Object | Size |
|---|---|
| Primary grid | 128 languages x 300 sentences x d coordinates (d = 4,096 for Qwen3-8B), one grid per layer |
| Extended dump | 128 x 1,500 x d at layer 0 and the dip layer, 14 models |
| FLORES replication | 116 x 300, 17 models |
| Content-transfer test | 19 models x 41 context languages x 100 pairs x 2 contexts, 157,700 scored sequences |
| Pooled benchmark panel | 33 models, 73 languages, 2,001 model-language rows before covariate filtering; 1,740 rows over 70 languages analyzed |
| Training interventions | Qwen3-0.6B-Base, 400 steps per condition, three seeds |

## What is not redistributed

The corpora themselves (NTREX-128 text, FLORES-200 text, Belebele, INCLUDE), the hidden-state dumps,
and model checkpoints. `data/NTREX/` holds only the language list, the document ids, and the license.
