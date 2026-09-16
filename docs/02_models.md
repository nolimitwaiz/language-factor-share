# Models

All models are public Hugging Face checkpoints, run frozen (no weight updates) except in the training interventions of `docs/05_training_interventions.md`. Inference ran in fp16 or bf16 as listed; hidden-state pooling and every statistic are computed in fp32. BLOOM is never run in fp16.

## Primary set: 17 base models, direct sums of squares

The registry below is `rmfs/configs/models.yaml`, which was generated from the stored results rather than typed by hand. `dip_layer` is the layer with the smallest LFS-VC under the frozen layer rule; the direct-SS minimum layer per model is in `rmfs/results/tables/lfs_depth_profile_audit.csv` and Table 5 of the paper. `campaign_dump` marks the 14 models with a 1,500-sentence hidden-state dump on the cluster, used by the resampling intervals and the misalignment decomposition.

| Tag | Hugging Face id | Blocks | Layers in the stored grid | Dip layer (registry) | Inference precision | 1,500-sentence dump |
|---|---|---|---|---|---|---|
| EuroLLM-1.7B | `utter-project/EuroLLM-1.7B` | 24 | 25 | 14 | fp16 | yes |
| Falcon3-7B-Base | `tiiuae/Falcon3-7B-Base` | 28 | 29 | 13 | fp16 | yes |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` | 32 | 33 | 14 | fp16 | no |
| Mistral-7B-v0.3 | `mistralai/Mistral-7B-v0.3` | 32 | 33 | 2 | fp16 | yes |
| OLMo-2-0425-1B | `allenai/OLMo-2-0425-1B` | 16 | 17 | 9 | fp16 | yes |
| OLMo-2-1124-7B | `allenai/OLMo-2-1124-7B` | 32 | 33 | 12 | fp16 | yes |
| Qwen3-0.6B-Base | `Qwen/Qwen3-0.6B-Base` | 28 | 29 | 8 | fp16 | yes |
| Qwen3-1.7B-Base | `Qwen/Qwen3-1.7B-Base` | 28 | 29 | 13 | fp16 | yes |
| Qwen3-4B-Base | `Qwen/Qwen3-4B-Base` | 36 | 37 | 17 | fp16 | yes |
| Qwen3-8B-Base | `Qwen/Qwen3-8B-Base` | 36 | 37 | 19 | fp16 | yes |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B` | 24 | 25 | 15 | fp16 | yes |
| Yi-1.5-9B | `01-ai/Yi-1.5-9B` | 48 | 49 | 20 | fp16 | no |
| bloom-1b7 | `bigscience/bloom-1b7` | 24 | 25 | 14 | fp16 | yes |
| bloom-7b1 | `bigscience/bloom-7b1` | 30 | 31 | 17 | fp16 | yes |
| granite-3.1-8b-base | `ibm-granite/granite-3.1-8b-base` | 40 | 41 | 14 | fp16 | no |
| salamandra-2b | `BSC-LT/salamandra-2b` | 24 | 25 | 6 | fp16 | yes |
| salamandra-7b | `BSC-LT/salamandra-7b` | 32 | 33 | 18 | fp16 | yes |

Family grouping follows released series (Qwen3 and Qwen2.5 are separate families; Tower, Occiglot, and Sailor are separate from their base families): 11 families in the primary set, 17 across the union of 33 models.

## Expansion set: 19 models, LFS-VC

Measured under the content-transfer protocol (`prereg/addenda/AIM1_R1_CONTENT_TRANSFER_PROTOCOL_2026-08-24.md`) at the pinned revisions below (`manifests/R1_A9_MODEL_REVISION_LOCK_2026-08-24.tsv`). Three of them (Falcon3-7B, SmolLM2-1.7B, Llama-3.1-8B, by family) overlap the primary set; the union is 33 distinct models.

| Tag | Hugging Face id | Revision | Family |
|---|---|---|---|
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | `060db6499f32` | Qwen2.5 |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | `8faed761d45a` | Qwen2.5 |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B` | `3aab1f1954e9` | Qwen2.5 |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | `d14972939875` | Qwen2.5 |
| xglm-1.7B | `facebook/xglm-1.7B` | `d23a5e8e2164` | XGLM |
| xglm-2.9B | `facebook/xglm-2.9B` | `33c659ae27de` | XGLM |
| xglm-7.5B | `facebook/xglm-7.5B` | `732d59308a84` | XGLM |
| mGPT | `ai-forever/mGPT` | `40897bd7c8b4` | mGPT |
| OLMo-2-1124-13B | `allenai/OLMo-2-1124-13B` | `3fefddc1bf18` | OLMo-2 |
| granite-3.1-8b-base | `ibm-granite/granite-3.1-8b-base` | `39975ba90995` | Granite |
| Yi-1.5-9B | `01-ai/Yi-1.5-9B` | `80d5471b1eae` | Yi-1.5 |
| TowerBase-7B | `Unbabel/TowerBase-7B-v0.1` | `bd59c10d7600` | Tower |
| occiglot-7b-eu5 | `occiglot/occiglot-7b-eu5` | `72d8b932ce09` | Occiglot |
| Sailor-1.8B | `sail/Sailor-1.8B` | `c687a96d5c3e` | Sailor |
| Sailor-7B | `sail/Sailor-7B` | `b8b49a0f0207` | Sailor |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` | `d04e592bb4f6` | Llama-3.1 |
| SmolLM2-360M | `HuggingFaceTB/SmolLM2-360M` | `f8027fd0eaee` | SmolLM2 |
| Falcon3-3B-Base | `tiiuae/Falcon3-3B-Base` | `092c29e3114f` | Falcon3 |
| Mistral-Nemo-Base-2407 | `mistralai/Mistral-Nemo-Base-2407` | `a4477a2f9779` | Mistral |

## Instruction-tuned pairs (control comparison)

Six base and instruction-tuned pairs on the same NTREX grid, direct sums of squares: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B (base against the Qwen3 instruct release of the same size), Qwen2.5-7B against Qwen2.5-7B-Instruct, Llama-3.1-8B against Llama-3.1-8B-Instruct. Protocol and predictions: `prereg/addenda/AIM1_EABC_PROTOCOL_2026-09-05.md` (E-C). Results: `results/instruct_grid/`, `results/eabc_scoring/`.

## Training model

All training interventions fine-tune `Qwen/Qwen3-0.6B-Base` (see `docs/05_training_interventions.md`).

## Revisions

The expansion set is pinned to the revisions above. The primary-set runs recorded their snapshot revisions in the cluster run ledger (`ledgers/run_ledger.jsonl`, one JSON record per run with code hash, configuration, input hashes, seed, host, precision, and wall time); the paper lists them in the camera-ready.
