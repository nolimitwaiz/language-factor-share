# Literature and novelty table for the LFS paper

**Date:** 2026-09-05 (first pass; entries marked VERIFY need a primary-source check
before the paper cites them).
**Purpose:** the ICLR plan and the project rules (`00_research_rules.md`) forbid any priority claim before this table
exists. For each neighbor: what it measures, on what, how it was validated, and the
exact distinction from LFS. The paper claims the narrowest novelty this table supports.

## The narrowest defensible novelty statement

> LFS is, to our knowledge, the first intrinsic multilinguality reading that
> (i) partitions representation variance between language and matched meaning as
> an exact balanced two-way decomposition with a closed-form null, (ii) is
> profiled across 33 open decoder LLMs, (iii) is validated against a
> confound-removed behavioral target on 19 models with model-level uncertainty,
> and (iv) has its failure modes measured by calibrated fault injection and
> trained intervention.

Not claimed: first to decompose representations (ANOVA is old; Chang et al. and
Libovický et al. subtract language means), first to observe middle-layer sharing
(Shani and Basirat, Tezuka and Inoue, Wendler et al., Kudugunta et al., Muller et
al., Tamo et al. 2026), first intrinsic metric correlated with downstream tasks (MEXA, Information
Parity), first to study cross-lingual linear maps (Mikolov, Conneau, Artetxe, Peng
and Søgaard).

## Intrinsic multilinguality metrics

| Work | Quantity | Data / models | Validation | Distinction from LFS |
|---|---|---|---|---|
| Kargaran et al. 2025, MEXA (Findings ACL) | fraction of parallel sentences whose translation is the mutual nearest neighbor (row and column dominance) against an English pivot, per layer | FLORES-200 and Bible; nine open models | raw Pearson with Belebele, m-MMLU, m-ARC up to about 0.9 | nearest-neighbor discriminability, English pivot, N-sensitive, no null, no confound control. LFS: variance partition, pivot-free, analytic null, deflated validation. Both blind to uniform collapse (shown here). |
| Tsvetkov and Kipnis 2024, Information Parity (Findings EMNLP) | NLL of a text in language L divided by NLL of its English counterpart | parallel text; several LLMs | correlation with multilingual task accuracy | output-behavior compression ratio, not a representation reading; not layerwise; English reference. |
| Libovický, Rosa, Fraser 2020 (Findings EMNLP; verified) | language centroids of mBERT representations; centering by language centroid yields more language-neutral representations | mBERT, XLM-R; parallel and retrieval probes | retrieval and language-ID probing | the closest conceptual ancestor: they remove language means; LFS quantifies how much variance those means carry, with a null, per layer, on decoder LLMs at scale, and validates the reading. |
| Chang, Tu, Bergen 2022 (EMNLP), geometry of multilingual representations | language-sensitive versus language-neutral axes via language means and PCA in XLM-R | XLM-R; 88 languages | geometric analyses; no downstream validation | descriptive geometry on an encoder; LFS turns the same additive picture into a validated scalar profile across many decoder LLMs. |
| Kudugunta et al. 2019 (EMNLP-IJCNLP, pp. 1565-1575; verified) | SVCCA similarity of NMT encoder representations across languages and layers | multilingual NMT (103 languages) | qualitative; language-family clustering | similarity index on an NMT encoder; not a variance share; no null; no downstream validation. |
| Kornblith et al. 2019, CKA (ICML) | representational similarity index, invariant to orthogonal maps and scale | general | n/a | comparator in our fault battery; invariant to the structure LFS measures. |

## Layerwise sharing and the "English" debate

| Work | Claim | Method | Relation to LFS |
|---|---|---|---|
| Wendler et al. 2024 (ACL) | Llama "works in English" at middle layers | logit lens on intermediate states | we measure the lens's validity limit (max cosine about 0.10 mid-stack) and find the shared factor beats English for 113 to 124 of 127 languages. |
| Shani and Basirat 2025 (BlackboxNLP) | middle layers are shared or language-specific spaces, not internal translation | representational similarity across layers | qualitative agreement; LFS supplies the number and the cross-model spread. |
| Tezuka and Inoue 2025 (EMNLP) | transfer neurons move between shared and language-specific spaces | neuron analysis | mechanism-level; LFS is population-level and does not claim neurons (basis sensitivity shown). |
| Zhao et al. 2024 (NeurIPS) | language-specific neurons concentrate in final layers | neuron ablation | consistent with the rise of LFS toward the output. |
| Xu et al. 2025 (EMNLP) | linguistic neuron overlap and transfer | neuron overlap | as above. |
| Bafna et al. 2025 (IJCNLP-AACL) | task-solving then implicit translation; middle layers multilingual with English foremost | lens and generation analysis | our hub result quantifies "multilingual with English foremost": English is the best single donor mostly for high-resource targets. |
| Dumas et al. 2025 (ACL) | language and concept separable causally | activation patching | causal counterpart to the observational decomposition; we cite, do not replicate. |
| Tamo et al. 2026, LinguaMap (ICLR 2026; verified) | per-layer language probability via logit lens plus mean-pooled cross-lingual cosine similarity | Qwen-3-32B, Qwen-3-8B, BLOOM-7.1B; six languages | language consistency after selective fine-tuning of late layers (98 percent) | closest accepted neighbor: same align, reason, generate reading; two to three models, six languages, no null, no cross-corpus replication, no behavioral validation of the layer reading; ends on a fine-tuning method. LFS: a variance share with a null on 33 models and up to 128 languages, replicated, deflation-validated, no fine-tuning. |
| Muller et al. 2021 (EACL; verified) | mBERT first aligns, then predicts; lower layers language-neutral for alignment | probing across layers | same shape story on an encoder; LFS profiles it on decoders with a null. |

## Mapping between language spaces

| Work | Finding | Relation |
|---|---|---|
| Mikolov et al. 2013; Conneau et al. 2018 (ICLR); Artetxe et al. 2018 | word-embedding spaces align by a linear or orthogonal map | the question the proposal asks of deep layers; our ladder answers: mostly offset plus scale, rotation under one percent. |
| Marchisio et al. 2020 to 2022 (Koehn's group) | when and why embedding alignment works; isomorphism | background the PI knows; cite in the report, third person in the paper. |
| Peng and Søgaard 2024 (EMNLP) | high-quality linear alignments between concept spaces in larger LLMs | supports our linear-map finding at the concept level; ours is held-out, nested, with permutation nulls and a rotation share. |
| Zhao et al. 2025, Lens (arXiv 2410.04407) | SVD split into language-agnostic and language-specific subspaces, used as training signal | decompose-to-intervene; we decompose-to-measure and show the intervention trap. |

## Alignment objectives (Aim 2 landscape; cited, not competed with)

| Work | Method | Relation |
|---|---|---|
| Liu and Niehues 2025 (ACL) | middle-layer alignment objective in task fine-tuning | our stress test shows scale-invariant alignment scores can rise while raw concept variance collapses; their gains are on tasks, which we do not dispute. |
| Bu et al. 2025, AlignX (EMNLP) | representation alignment plus language-adversarial term | same caveat. |
| Briakou, Cherry, Foster 2023 (ACL) | incidental bilingualism drives translation ability | background for the data-exposure confound. |

## Confounds, anisotropy, and evaluation methodology

| Work | Finding | Relation |
|---|---|---|
| Timkey and van Schijndel 2021 (EMNLP); Ethayarajh 2019 (EMNLP) VERIFY; Sun et al. 2024, massive activations (arXiv) VERIFY | a few rogue dimensions dominate similarity; representations are anisotropic | why LFS standardizes jointly, why whitening changes the picture, why fp32 pooling. |
| Lin et al. 2019 LangRank (ACL); Xia et al. 2020 NLPerf (ACL) | regress task performance on language features | the regression form we reuse to deflate rather than predict. |
| Ahia et al. 2023 (EMNLP); Petrov et al. 2023 (NeurIPS) | tokenizer fertility unfairness | fertility as covariate and as representation-side finding. |
| Nosek et al. 2018 (PNAS) | preregistration | the discipline behind the frozen predictions. |

## Datasets

| Work | Role |
|---|---|
| Federmann, Kocmi, Xin 2022, NTREX-128 | representation grid |
| NLLB team 2022, FLORES-200 (arXiv 2207.04672) VERIFY citation form | replication corpus (E-B) |
| Bandarkar et al. 2024, Belebele (ACL) | translated downstream target |
| Romanou et al. 2025, INCLUDE (ICLR) | native downstream control |
| Nguyen et al. 2023, CulturaX | exposure proxy |

## What still needs a search before submission

1. Done 2026-09-05: Libovický et al. 2020 (Findings EMNLP 2020; unsupervised per-language centering yields more language-neutral representations), Kudugunta et al. 2019 (EMNLP-IJCNLP 2019; SVCCA, 103 languages), Muller et al. 2021 (EACL 2021) verified and added to the paper's bibliography and related work.
2. Add Ethayarajh 2019 (EMNLP) and Sun et al. 2024 (massive activations) to the bib with exact titles if space allows.
3. Search "variance decomposition" and "ANOVA" with "multilingual representations" for any direct precedent of a language-versus-concept variance share; if one exists, cite it and narrow the novelty statement further.


## Addendum, 7 September 2026: second sweep

Searches run on 7 September 2026 for variance decompositions of language versus content in multilingual hidden states, language-share or sum-of-squares readings per layer, language-centroid follow-ups, and the exact phrase "language factor share". No paper computes a per-layer share of language versus matched-content variance with a closed-form null. One lead was checked and dismissed: a search summary claimed that Tezuka and Inoue (2025) use a between/within-language correlation ratio; the full text uses PCA, SVD explained-variance ratios, cosine and Euclidean distances, and a mutual k-NN alignment metric, not a correlation ratio.

| Neighbor | What it measures | Overlap with LFS | What LFS adds |
|---|---|---|---|
| Bayazit, AlKhamissi, Bosselut 2026 (arXiv 2609.00155) | geometry-based vs decoding-based latent-language probes disagree; up to 27 languages | shares the point that decoding probes are not representation probes | a share with a null instead of a language label |
| Sakajo et al. 2026 (arXiv 2606.01800) | per-layer spanning-tree structure and tree edit distance between languages on FLORES+ parallel sentences; 3 models, 8 languages | per-layer, parallel sentences, structural distance to English | no English pivot, 33 models, 128 languages, behavioral validation |
| Marinov et al. 2026 (arXiv 2606.14347) | languages as approximately separable linear factors under a covariance-adjusted inner product; 3 models, 28 bilingual contrasts | additive/separable language factor | a variance share per layer, null, replication, behavior |
| Ravisankar et al. 2025 (arXiv 2504.09378) | DALI instance-level alignment to English at middle layers predicts NLU errors; patching | validation against behavior | no pivot; confound-controlled; model-level generalization |
| Wilie et al. 2025 (arXiv 2503.11280) | Interlingual Local Overlap across layers, 31 languages | per-layer alignment consistency | variance partition rather than neighborhood overlap |
| Liang, Dufter, Schütze 2021 (arXiv 2109.08040) | dimensionality and layers of language-specific information; linear subspace | language-specific subspace per layer | share of systematic variance against matched content |
| Xie et al. EMNLP 2022 | low-rank language-specific subspace by SVD, projected away | subspace removal | measurement rather than removal; null |
| Hämmerl, Libovický, Fraser, Findings ACL 2024 | survey of alignment notions | framing | cited for the language-neutral vs language-specific trade-off |
| Li et al. AAAI 2025 (Language Ranker) | similarity to English as a per-language performance proxy | intrinsic metric vs performance | no pivot; deflation for exposure |
| Wu et al. ICLR 2025 (Semantic Hub) | shared space across languages and modalities | middle-layer sharing | quantified per layer with a null |

All ten are now cited in the paper's related work and the technical report. The narrowest defensible novelty statement is unchanged: LFS is the first per-layer share of language versus matched-content variance with a closed-form null, measured on 33 models and up to 128 languages, replicated on a second corpus, validated against behavior under confound control, and characterized under calibrated fault injection, to the extent of two literature sweeps (July and September 2026). The paper does not use the phrase "first of its kind".


## Second sweep, 7 September 2026: 66 verified works added to the paper (Appendix: Extended related work) and the report (Section 2)

Every entry was opened on its canonical page (arXiv abstract, ACL Anthology, or publisher) and its BibTeX checked before insertion. Two sentences that mis-stated the model count were corrected (33 models, 14 families) before insertion.

| key | title | venue | cite where |
|---|---|---|---|
| gonen2020greek | It's not Greek to mBERT: Inducing Word-Level Translations from Multilingual BERT | Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP | Paper: Related work (language-neutral vs language-specific components) and Stress tests (misalignment budget) |
| zhao2021inducing | Inducing Language-Agnostic Multilingual Representations | Proceedings of *SEM 2021: The Tenth Joint Conference on Lexical and Computational Semantics | Paper: Related work; Limitations (LFS is a reading, not a repair) |
| kojima2024neurons | On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons | Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) | Paper: Related work; Results discussion of the interior minimum with endpoint recovery |
| zhao2025disentanglement | When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners | Advances in Neural Information Processing Systems (NeurIPS 2025) | Paper: Related work; Stress tests (mean-shift and subspace interventions at intermediate layers) |
| lopo2025surgery | Language Surgery in Multilingual Large Language Models | Proceedings of the 5th Workshop on Multilingual Representation Learning (MRL 2025) | Paper: Related work; Results (interior minimum with endpoint recovery) |
| zhong2025think | What Language Do Non-English-Centric Large Language Models Think in? | Findings of the Association for Computational Linguistics: ACL 2025 | Paper: Related work and the results paragraph on dip depth versus training recipe; Report: Section 2 |
| deng2025unveiling | Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders | Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Related work and Stress tests (misalignment budget decomposition) |
| zeng2025converging | Converging to a Lingua Franca: Evolution of Linguistic Regions and Semantics Alignment in Multilingual Large Language Models | Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025) | Paper: Related work and the Discussion of dip depth versus training recipe; Report: Section 2 |
| cheng2025abstraction | Emergence of a High-Dimensional Abstraction Phase in Language Transformers | Proceedings of the Thirteenth International Conference on Learning Representations (ICLR 2025) | Paper: Related work (representation geometry across layers) and Discussion of the interior minimum |
| lauscher2020hero | From Zero to Hero: On the Limitations of Zero-Shot Language Transfer with Multilingual Transformers | Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) | Paper: Related work; Stress tests (confound removal); Limitations |
| ali2024tokenizer | Tokenizer Choice For LLM Training: Negligible or Crucial? | Findings of the Association for Computational Linguistics: NAACL 2024 | Paper: Stress tests (confound removal) and Limitations; Report: Section 13 covariate discussion |
| bagherinezhad2024drives | What Drives Performance in Multilingual Language Models? | Proceedings of the Eleventh Workshop on NLP for Similar Languages, Varieties, and Dialects (VarDial 2024) | Paper: Setup (covariates) and Stress tests (confound removal); Report: Section 13 covariate list |
| wu2025bitter | The Bitter Lesson Learned from 2,000+ Multilingual Benchmarks | arXiv preprint arXiv:2504.15521 | Paper: Related work and Stress tests (pooled exam null); Report: Section 13 discussion of exam reliability |
| idris2026embedding | Can Embedding Similarity Predict Cross-Lingual Transfer? A Systematic Study on African Languages | arXiv preprint arXiv:2601.03168 | Paper: Stress tests (model-bootstrap interval, family clustering) and Limitations; Report: Section 13 uncertainty discussion |
| rudman2022isoscore | IsoScore: Measuring the Uniformity of Embedding Space Utilization | Findings of the Association for Computational Linguistics: ACL 2022 | Paper: Method (standardization) and Stress tests or Limitations (sensitivity to preprocessing) |
| hammerl2023anisotropy | Exploring Anisotropy and Outliers in Multilingual Language Models for Cross-Lingual Semantic Sentence Similarity | Findings of the Association for Computational Linguistics: ACL 2023 | Paper: Related work; Method (choice of standardization); Stress tests (collapse blind spot, retrieval versus CVP) |
| ding2021grounding | Grounding Representation Similarity Through Statistical Testing | Advances in Neural Information Processing Systems 34 (NeurIPS 2021) | Paper: Related work and Stress tests (validation design) |
| kriegeskorte2019onion | Peeling the Onion of Brain Representations | Annual Review of Neuroscience | Paper: Related work (variance-partition analyses of representations in other fields) |
| chi2021infoxlm | InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training | Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies | Paper: Related work (training objectives) and Method (definition of the contrastive arm); Report: Section 13 setup |
| bardes2022vicreg | VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning | Proceedings of the Tenth International Conference on Learning Representations (ICLR 2022) | Paper: Stress tests (bounded CVP hinge) and Limitations; Report: Section 13 on the CVP penalty arm |
| ji2024emma | EMMA-500: Enhancing Massively Multilingual Adaptation of Large Language Models | arXiv preprint arXiv:2409.17892 | Paper: Related work (continued pretraining for low-resource languages) and Limitations or future work; Report: Section 13 alternatives to the proposal's loss |
| gao2023scaling | Scaling Laws for Reward Model Overoptimization | Proceedings of the 40th International Conference on Machine Learning (ICML 2023) | Paper: Stress tests (metric-as-loss reversal) and Limitations; Report: Section 13 interpretation |
| conneau2020emerging | Emerging Cross-lingual Structure in Pretrained Language Models | Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics | Paper: Related work |
| tiyajamorn2021representation | Language-agnostic Representation from Multilingual Sentence Encoders for Cross-lingual Similarity Estimation | Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing | Paper: Related work; Method (motivation for the additive decomposition) |
| huang2024erasure | Language Concept Erasure for Language-invariant Dense Retrieval | Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing | Paper: Stress tests (metric-as-loss reversal; collapse blind spot); Limitations |
| sterz2025recover | ReCoVeR the Target Language: Language Steering without Sacrificing Task Performance | Findings of the Association for Computational Linguistics: EMNLP 2025 | Paper: Related work; Stress tests (misalignment budget: offset dominance explains why mean-difference steering works) |
| kim2026langsae | LangSAE Editing: Improving Multilingual Information Retrieval via Post-hoc Language Identity Removal | arXiv preprint arXiv:2601.04768 | Paper: Related work (recent post-hoc removal methods); Limitations |
| tang2024neurons | Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models | Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Related work (language-specific neurons) and the Results paragraph introducing the interior minimum with endpoint recovery |
| harrasse2025transcoders | Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders | arXiv preprint arXiv:2511.10840 | Paper: Related work and Limitations (early-layer readings depend on whether one measures feature overlap or variance share) |
| korner2026meanings | When Meanings Meet: Investigating the Emergence and Quality of Shared Concept Spaces during Multilingual Language Model Training | Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Related work and Limitations (LFS covers final checkpoints, not training dynamics); Report: Section 13 |
| skean2025layer | Layer by Layer: Uncovering Hidden Representations in Language Models | Proceedings of the 42nd International Conference on Machine Learning | Paper: Related work and Setup (justification for the dip-layer readout used in the R_content validation) |
| artetxe2020artifacts | Translation Artifacts in Cross-lingual Transfer Learning | Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) | Paper: Setup (data, matched-translation grid) and Limitations (translationese); Report: data section |
| gaschi2023alignment | Exploring the Relationship between Alignment and Cross-lingual Transfer in Multilingual Transformers | Findings of the Association for Computational Linguistics: ACL 2023 | Paper: Related work (intrinsic measures correlated with downstream performance); Report: Section 2 origin and prior validations |
| singh2025globalmmlu | Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation | Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Stress tests (pooled exam null) and Limitations; Report: Section 13 discussion of exam criteria |
| miller2024evals | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations | arXiv preprint arXiv:2411.00640 | Paper: Setup (statistics) and Stress tests (uncertainty); Report: bootstrap methodology |
| ethayarajh2019contextual | How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings | Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) | Paper: Related work (geometry of hidden states) and Method (why standardize before partition) |
| kovaleva2021outlier | BERT Busters: Outlier Dimensions that Disrupt Transformers | Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021 | Paper: Method (standardization) and Setup |
| sun2024massive | Massive Activations in Large Language Models | First Conference on Language Modeling (COLM) | Paper: Method (standardization) and Setup (pooling and layer selection) |
| hewitt2019control | Designing and Interpreting Probes with Control Tasks | Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) | Paper: Method (noise reference value) and Related work (probing critiques) |
| cao2020alignment | Multilingual Alignment of Contextual Word Representations | International Conference on Learning Representations (ICLR) | Paper: Related work and Stress tests (collapse blind spot); Report: Section 13 (word-alignment arm) and Section 2 (Origin of the Idea) |
| wang2020uniformity | Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere | Proceedings of the 37th International Conference on Machine Learning | Paper: Stress tests (collapse blind spot) and Method (why raw-variance CVP is needed); Report: Section 13 analysis of scale invariance |
| conneau2020unsupervised | Unsupervised Cross-lingual Representation Learning at Scale | Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics | Paper: Related work (alignment versus capability trade-offs) and Limitations; Report: Section 13 code-switched arm |
| dasilva2025steering | Steering off Course: Reliability Challenges in Steering Language Models | Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Related work (interventions) and Limitations; Report: Section 13 discussion of intervention alternatives to the proposal's loss |
| yan2023bleurt | BLEURT Has Universal Translations: An Analysis of Automatic Metrics by Minimum Risk Training | Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Stress tests (why LFS is a measurement, not a loss) and Limitations; Report: Section 13 and Section 2 |
| yang2021bias | A Simple and Effective Method To Eliminate the Self Language Bias in Multilingual Representations | Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing | Paper: Related work; Method (why LFS applies joint z-scoring rather than component truncation before measuring) |
| rajaee2022isotropy | An Isotropy Analysis in the Multilingual BERT Embedding Space | Findings of the Association for Computational Linguistics: ACL 2022 | Paper: Setup or Method (normalization choice), alongside the existing rogue-dimensions citation |
| schut2025english | Do Multilingual LLMs Think In English? | arXiv preprint arXiv:2502.15603 | Paper: Related work; Stress tests (English-hub test) |
| zhong2026sparse | Language Lives in Sparse Dimensions: Toward Interpretable and Efficient Multilingual Control for Large Language Models | Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Related work; Limitations (LFS treats coordinates symmetrically after z-scoring and does not localize dimensions) |
| belrose2023tuned | Eliciting Latent Predictions from Transformers with the Tuned Lens | arXiv preprint arXiv:2303.08112 | Paper: Related work (layerwise readouts) and Method (why LFS uses no fitted probe or vocabulary projection); Report: Section 2 Origin of the Idea |
| zhang2024regions | Unveiling Linguistic Regions in Large Language Models | Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Related work; Report: Section 13 (Aim 2 campaign, region- and layer-selective training options) |
| brinkmann2025grammatical | Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages | Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) | Paper: Related work (interlingua and shared-concept claims); Report: Section 2 |
| bandarkar2025swapping | Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models | The Thirteenth International Conference on Learning Representations (ICLR) | Paper: Related work (layer-selective fine-tuning); Report: Section 13 (Aim 2 alternatives) |
| wu2020equal | Are All Languages Created Equal in Multilingual BERT? | Proceedings of the 5th Workshop on Representation Learning for NLP | Paper: Related work (confounds) and Stress tests (confound removal for the pooled exam null); Report: Section 13 covariate discussion |
| rust2021tokenizer | How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models | Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) | Paper: Setup (covariates) and Stress tests (confound removal); Report: Section 13 covariate list |
| wang2024emergence | Probing the Emergence of Cross-lingual Alignment during LLM Training | Findings of the Association for Computational Linguistics: ACL 2024 | Paper: Related work; Limitations (checkpoint-time dependence); Report: Section 13 discussion of recipe versus size |
| kreutzer2025dejavu | Déjà Vu: Multilingual LLM Evaluation through the Lens of Machine Translation Evaluation | Proceedings of the Second Conference on Language Modeling (COLM) | Paper: Setup (evaluation protocol) and Limitations; Report: methodology section and Section 13 protocol |
| alali2026english | Predicting Multilingual Classification and Translation Performance of LLMs with Cross-Lingual Alignment -- Is English Enough? | arXiv preprint arXiv:2608.03446 | Paper: Related work (intrinsic measures vs |
| godey2024anisotropy | Anisotropy Is Inherent to Self-Attention in Transformers | Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Related work; Method (standardization rationale) |
| puccetti2022outlier | Outlier Dimensions that Disrupt Transformers are Driven by Frequency | Findings of the Association for Computational Linguistics: EMNLP 2022 | Paper: Method (standardization) and Limitations (frequency and tokenization confounds across languages) |
| davari2023cka | Reliability of CKA as a Similarity Measure in Deep Learning | The Eleventh International Conference on Learning Representations (ICLR) | Paper: Related work (similarity measures) and Stress tests (measurement under optimization pressure) |
| belinkov2022probing | Probing Classifiers: Promises, Shortcomings, and Advances | Computational Linguistics | Paper: Related work and Limitations |
| wu2020explicit | Do Explicit Alignments Robustly Improve Multilingual Encoders? | Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) | Paper: Related work and Limitations; Report: Section 13 discussion of why alignment losses did not help |
| jing2022collapse | Understanding Dimensional Collapse in Contrastive Self-supervised Learning | The Tenth International Conference on Learning Representations (ICLR) | Paper: Stress tests (collapse blind spot) and Limitations; Report: Section 13 interpretation of the word-alignment arm |
| chang2024curse | When Is Multilinguality a Curse? Language Modeling for 250 High- and Low-Resource Languages | Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing | Paper: Related work and Limitations (what LFS cannot say about data mixture); Report: Section 13 discussion of unseen-language cost |
| sundar2025steering | Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models | Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Related work (interventions and alignment readings) and Stress tests (collapse blind spot); Report: Section 13 on retrieval readings |
| she2024mapo | MAPO: Advancing Multilingual Reasoning through Multilingual-Alignment-as-Preference Optimization | Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) | Paper: Related work (training objectives) and Limitations; Report: Section 13 discussion of the preference-training sub-aim |
