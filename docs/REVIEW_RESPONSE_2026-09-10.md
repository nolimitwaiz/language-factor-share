# Response to the 10 September 2026 review of the ICLR draft

Source: the external review pasted by Waiz on 2026-09-10 (17 numbered items plus a
second message with a language table, figure and structure notes, and the ICLR
2027 requirements). Every item below says whether the change was necessary for
correctness, recommended for clarity, or optional, and what was done in
`paper/iclr2027/main.tex`, `abstract_for_openreview.txt`, and `references.bib`.
Numbers were not changed; no experiment was run.

| # | Review item | Necessary? | Done |
|---|---|---|---|
| 1 | Keep the title | n/a | Kept: "Language-Factor Share: Measuring and Stress-Testing Multilingual Representation Structure". |
| 2 | State the central claim consistently (LFS measures the relative contribution of language and matched-sentence main effects; characterized across layers and models; related to behavior; limits identified) | Necessary | Abstract, contributions (1) to (4), Section 5 headings, Section 6 title, Limitations, and Conclusion now say this and nothing stronger. "Semantic understanding", "internal reasoning language", and "what determines multilingual ability" wording removed. |
| 3 | "Concept" is an operational label | Necessary | Sentence added after the decomposition: the matched-sentence main effect is the content component, an operational proxy for structure shared across translations, not a direct measure of all semantic information. Formulas use "sentence main effect"; prose uses "content component" or "sentence-component share". "Concept direction" removed everywhere. |
| 4 | Make the denominator unmistakable | Necessary | Interpretation paragraph now states that LFS is a share of the two main effects, not of total variance, with the 2/8/90 example (LFS 0.20 while the sentence effect is 8% of the total). Extended related work sentences that said "total per-layer variance" or "fraction of standardized variance" now say "the two main-effect sums of squares". |
| 5 | LFS versus LFS-VC naming | Necessary | Abstract, contribution (3), Section 6 text, Table 2 rows and caption, Figure 2 caption and axis label (figure regenerated), Table 8 caption, and the expansion-set statements now name LFS-VC wherever it was used. Appendix C states that levels differ (Llama-3.1-8B minimum 0.626 direct SS versus 0.479 LFS-VC), that the estimators are never pooled, and that a same-protocol direct-SS re-measurement of the 19 expansion models has not been run. |
| 6 | Tighten the abstract | Recommended | Rewritten to 221 words in the order question, definition, structural result, behavioral evidence, limits and use; formula moved to Methods; every phrase in the review's table replaced as suggested; estimator named. |
| 7 | Describe the behavioral task precisely | Necessary | Setup item (iii) and the Section 6 paragraph now open with "whether context in another language improves the likelihood of an English continuation, relative to a mismatched-context control"; heading changed to "Foreign-language context and English continuation likelihood". |
| 8 | Replace causal and absolute language | Necessary | "With confounds removed" to "after adjustment for measured covariates"; "about half of that is exposure" to "decreases after adjustment for exposure-related covariates"; "set by training more than by size" to "varies across model families and training histories and is not explained by parameter count alone"; "property of pretraining" to "largely preserved in the six base and instruction-tuned comparisons"; "generalizes across models and families" to "remains positive in the reported model-level and family-deletion sensitivity analyses". Limitations now separate observational cross-model associations from the controlled training interventions on one model. |
| 9 | Minimum layer, minimum LFS, dip depth | Necessary | Defined as three terms in Profile statistics; SmolLM2 and Falcon3 sentence now reports dip depths; Figure 1 caption points to the table with minimum layer and value; "interior minimum" used instead of "middle layers" for the result. |
| 10 | Model counts and populations | Necessary | Primary set is eleven named families (was "nine"); grouping rule stated (released series; Qwen3 and Qwen2.5 separate; Tower, Occiglot, Sailor separate), giving 17 families across the union of 33 models (was "14"); the 33 models are one population only for the profile-shape claim; "macro-family" covariate identified as a language family; the "above 0.5" claim restricted to the primary set with the three LFS-VC minima below 0.5 listed; new Table 7 "Analysis populations" (models, languages, sentences, estimator, criterion per analysis). |
| 11 | Mathematical qualifications | Necessary | Null: stated as an expectation around which a realized noise grid scatters; the claim that "differences at a fixed grid are unaffected because noise is common" removed. Interactions and language-specific transformations: now "not identified separately; enter the residual and can also change the main-effect sums of squares". Rotation invariance: distinguished for an already standardized grid versus rotating raw states and re-standardizing (main text and new Appendix I). LFS 0 or 1: now stated not to imply that individual representations coincide. |
| 12 | Shrinking versus dimensional collapse versus harm | Necessary | "Uniform representation collapse" to "uniform shrinking of representation magnitude"; the paragraph states that nonzero uniform scaling changes neither rank nor the relative spectrum and is distinct from dimensional collapse; the trained-model paragraph reports exactly what was observed (norms and raw content variance fell, retrieval scores rose) and keeps the Belebele qualification: the check flags representation change, not demonstrated capability loss. |
| 13 | Negative findings without explaining them away | Necessary | The "no predictor can exceed about 0.56" ceiling removed; the translationese versus resource-range comparison now "points to" rather than proves; the pooled-benchmark result stated as "no clear adjusted association on this panel; limits the interpretation of LFS as a general performance indicator"; "sentence bootstrap" renamed "sentence-subsampling intervals ... drawn without replacement ... at the fixed stored layers" in Section 5, Table 9 caption, and Table 7. |
| 14 | Bound the mapping and hub conclusions | Necessary | Headings now "Offsets and scale explain most measured misalignment" and "A multilingual average predicts target-language representations better than an English donor"; "evidence of absence" replaced by "additional orthogonal alignment provides little held-out improvement under this protocol" (paper and technical report). |
| 15 | Related work: selective, less defensive | Recommended | "What the field lacks is a measurement" removed. Added and discussed: Shah et al. (2024, LREC-COLING), the Koehn-coauthored multilingual emotion-neuron paper (Zhao, Koehn, Schuller, Sisman 2026), Poelman and de Lhoneux (2026), DEPART (2026), Hu, Vulić, Korhonen (2025), Chirkova and Nikoulina (2024), Ki et al. (2025), plus the proposal's own sources for mapping and hubness (Marchisio et al. 2021, 2022; Koehn and Knight 2002; Artetxe et al. 2017; Radovanović et al. 2010), language neurons after alignment (Zhang et al. 2026), benchmark problems (Deng et al. 2024; Gema et al. 2025), and Ding and Koehn (2021). Finance sources kept brief and moved to the appendix. Extended related work intro no longer says "none is a ranking". |
| 16 | Stop repeatedly defending the diagnostics' status | Recommended | One statement remains ("We evaluate it alongside an external retrieval-based measure, MEXA, and complementary diagnostics ... each reported separately"); "not ranked", "not part of the family", and "none averaged into another" removed from the abstract, contributions, Table 1 caption, Table 2 caption, setup, and Section 6. MEXA is called a "reference measure" rather than a "baseline", per Waiz's standing instruction; its point estimate is reported beside LFS with overlapping intervals. |
| 17 | Simpler, consistent prose | Recommended | "out-dips" to "has a larger dip depth than"; "never-touched" to "held-out model families"; fertility defined as tokens per word at first use; "cheap" to "closed-form once representations are available"; metadata need stated (language membership and matched translations); American spelling; duplicated "1,500-sentence" removed; layer counting stated (layer 0 is the embedding output, B blocks give B+1 layers); "four scripts" corrected to three (Latin, Cyrillic, Arabic); "Aim 2" removed from the paper. |

Second message (language table and figures):

| Item | Done |
|---|---|
| "concept direction" | Replaced by "sentence-component share, 1 − LFS-VC" where that estimator was used and "1 − LFS" for the direct-SS panel. |
| "endpoint recovery" | Kept as a defined term ("interior minimum with endpoint recovery", defined in Profile statistics); abstract says "interior minimum with both endpoints higher". |
| "meaning-weighted mid-network" | Replaced by "lower at an interior layer" / "smaller language main effect relative to the sentence main effect". |
| "confounds removed", "once exposure is removed" | Replaced by "after adjustment for measured covariates" / "exposure-related covariates". |
| "homogeneous of degree zero" | Replaced by "unchanged when all representations are multiplied by the same nonzero constant". |
| "attribution ladder" | Already "nested sequence of maps". |
| "purpose-specific behavioral checks" | Replaced by "checks against the behavior it is meant to track". |
| "concept variance preservation" | Now "content-variance preservation (CVP: raw sentence-component variance relative to the starting checkpoint)". |
| Figure 1 | Caption distinguishes minimum layer, minimum value, and dip depth and points to the per-model table. Width kept at 0.66 of the line to hold the nine-page limit. |
| Computation diagram in main text | Not moved: with the added citations and qualifications the main text fills nine pages exactly; the schematic stays as Appendix Figure A.1 and is referenced from Methods. |
| Figure 2 | Axis label corrected to 1 − LFS-VC; Table 2 wrapped so it no longer exceeds the right margin. Enlarging the figure was traded against the page limit. |
| Ratio-versus-raw-variance figure | Kept in Appendix J with the Belebele qualification; not promoted for the same page reason. |
| Structure | The seven-question order (what is measured, how computed, on what, what repeats, relation to behavior, when it misleads, what it is for) maps onto the existing Sections 1, 3, 4, 5, 6, 7, and the Conclusion. |

ICLR 2027 requirements re-checked on 2026-09-10 (Author Guidelines, AI Policy for Authors, Call for Papers, ICLR blog of 2 September 2026): main text 9 pages at submission and 10 during discussion; references and appendices unlimited; the AI-use, ethics, and reproducibility statements do not count; abstract deadline 18 September 2026 11:59 pm AoE, paper 25 September; abstracts must be genuine; no authors added after the abstract deadline (order may change until the paper deadline); title and abstract editable until the paper deadline; at most 20 submissions per author; at most one submission without a qualified reciprocal reviewer; every submission needs one author registered to review at least three papers; mandatory AI-use section in the paper and on the form. The rebuilt paper: 35 pages, main text ends on page 9, abstract 221 words.

Not done, and why: the estimator-labeling fix does not remove the underlying limitation that the 19-model behavioral result rests on LFS-VC while the profile results rest on direct SS; a direct-SS re-measurement of those 19 models would need cluster time and is listed as open. The corrections-ledger summary (Appendix L), the anonymous code link, snapshot hashes, and the AI-use statement wording remain TODO items in the draft.

## Third review (six items), applied 10 September 2026, evening

Every claim in the review was checked against the stored artifacts before editing.
Items 2 to 5 were correct as stated. Item 1 was accepted as the safer wording.
Item 6 (the TODO items) exposed two number discrepancies that are now recorded
in `docs/DISCREPANCY_LEDGER.md` (D13, D14) and `paper/iclr2027/claims_ledger.csv`
(C62, C63).

| # | Review item | Done |
|---|---|---|
| 1 | Abstract: "reversed association when minimized as a training objective" | Replaced with "the tested training interventions did not establish that optimizing it improves downstream behavior". Introduction contribution (4) now says "retrieval-tail alignment". |
| 2 | Ordering claim in Section 5 | Now "the subset ordering agrees with the grid ordering at Spearman >= 0.90 in 2,000 of 2,000 subsets (median 0.996)" (artifact: results/eabc_scoring/summary.json, criterion P-A3). Same fix in the technical report, the study guide, and both decks. |
| 3 | Estimator labels and panel sizes in Section 6 and Table 7 | The 0.158 same-support result and the 0.041 pooled result are labeled 1 - LFS-VC. Table 7 now reads "70 of 73 languages, 1,740 of 2,001 rows, pooled Belebele and INCLUDE residual"; the text gives the analyzed counts and points to the table. |
| 4 | Model fixed effects | Methods now state that the content-transfer adjustment omits model fixed effects (one predictor value per model) and takes its uncertainty from the model bootstrap; the generic regression keeps them. |
| 5 | "ranked first among all conditions" / "scored best" | Section 7 now reports what was measured: MEXA rose by +0.09 over the LM-only control and the shrinking conditions had the highest retrieval and tail scores as a group. The introduction says "among the best". Same fix in the technical report (two places), the one-pager, the study guide, and both decks. |
| 6 | Remaining TODO items | Regression appendix filled from results/deflation/attribution/regression_summary.txt (17-model grid, n = 1,241, token coefficient +0.286 SE 0.059, fertility -0.103, R^2 = 0.824); the main text had quoted the 10-model development run (+0.267, R^2 0.86) and now reports the 17-model values (D13). Per-model decomposition table added (Table "Misalignment decomposition at each model's dip layer", from results/budget/*_fixed_raw/budget.json); building it showed that "65 to 98 percent" was the three-model prototype range and the 14-model common-rank range is 63 to 97 percent, corrected everywhere (D14). Corrections-ledger appendix written from D1 to D14. AI-use statement finalized. Code-link and revision sentences reworded as commitments for the camera-ready. Figure 2 enlarged from 0.34 to 0.44 of the line width; the FLORES, subsampling, and measurement-dependence paragraphs compressed to one or two sentences with appendix pointers (new Appendix "Measurement dependence"). Main text still ends on page 9. |

Not changed: the pipeline diagram stays in the appendix (the nine-page limit is
exact after the additions above). The direct-SS re-measurement of the 19 expansion
models remains open and needs cluster time.

Presentation: the teaching deck was cut from 32 to 27 slides. Kept as experiments:
17-model profiles, size versus family, the English-hub test, the misalignment
decomposition (two slides), the foreign-context test (two slides), the two-behavior
comparison, and the shrinking case. The second corpus, sentence resampling, sealed
predictions, injected faults, and instruction tuning are now one slide, "How the
measurement was checked", with each result stated. The training-objective result
is stated with its numbers on the limits slide. The concept slides use the new flat
diagrams (docs/study/figs/pipeline/c1 to c7) with real values.

Wording (Waiz, 10 September, late): "held-out families" replaced by "unseen model families" in every document; the cross-validation sense of held-out (sentences, error reduction) is unchanged. Failures now stated explicitly in the paper: a new appendix paragraph "The failed prediction" reports the one failed frozen prediction of twelve (precision bound 0.03 exceeded by Mistral-7B 0.045 and Salamandra-2B 0.059) and the granite-3.1-8B cross-corpus outlier (0.190 to 0.074), and the main text points to it.

Build note (10 September, late): the local tectonic build had been falling back to Latin Modern because the ICLR `times` package has no Unicode-encoding font shape under XeTeX; `\usepackage[T1]{fontenc}` now makes the local build use the same Nimbus Times fonts that Overleaf's pdfLaTeX uses. Under Times the paper is 34 pages and the main text ends on page 9 with about a quarter page to spare, which was used to enlarge Figure 2 to half the line width. An Overleaf-ready zip is in deliverables/2026-09-10.

## Fourth review (three statements), applied 11 September 2026

| # | Review item | Done |
|---|---|---|
| 1 | Page 9 whitening statement contradicted Appendix J | Now: pooling choice changes absolute dip depth by 0.13 to 0.20 while the ordering changes by at most one adjacent transposition; ZCA whitening changes the ordering substantially (Spearman 0.45). |
| 2 | "six contrastive objectives" | Now: "across 18 condition-seed points from six experimental conditions, LFS and tail alignment were positively correlated (Pearson +0.92)"; the six conditions include the frozen checkpoint and the LM-only control. |
| 3 | Appendix H rank agreement of 1.000 | Now states that the agreement concerns dip-depth rankings on the 14 models measured by both estimators and does not establish equivalence of the minimum-layer sentence-component shares used for the behavioral analysis, which is why that result is labeled LFS-VC throughout. |

Figures: all four paper figures redrawn in a plain style (one message per chart, two
colors, direct labels, Helvetica), produced by paper/iclr2027/figs/make_fig1.py,
make_fig_r1.py, make_fig15.py, make_fig17.py with figstyle.py. Figure 1(b) now shows
the 95 percent sentence-subsampling intervals.

## Fifth review (figures and notation), applied 12 September 2026

Figure 1 caption now matches the drawing (range band, median, Qwen3-8B; intervals in (b); no FLORES markers) and panel (a) is titled "every profile has an interior minimum". Figure 2 enlarged to 0.62 of the line width; caption states the plotted 0.77 versus the adjusted 0.51 and that gray lines are label connectors. Decomposition caption no longer says "annotated"; rotation shares are referenced to the per-model table. Definition now states LFS is defined when SS_lang + SS_con > 0. Pipeline diagram relabeled: layer axis 0 to K (K = depth, L stays the number of languages), dimension d with 4,096 given as the Qwen3-8B example. Title kept.

## Fifth review: source-level audit (15 September 2026), applied 15 September

Every claim was checked against the producing code and saved results before editing. No reported number changed.

| Finding | Verified? | Done |
|---|---|---|
| Estimator-agreement statistic mislabeled as dip-depth rankings | Yes: `rmfs/scripts/audit_lfs_headline_claims.py::estimator_parity` compares 1 − LFS against q_L at the saved dip layer of the 1,500-sentence grids | Rewritten in Section 5, Appendix (REML), and the content-transfer appendix as agreement of sentence-component shares at the saved dip layer |
| Noise value derivation gives a ratio of expectations | Yes | Appendix now states the chi-square/Beta argument (independent projections, exact expectation (L−1)/(L+N−2)) and that empirical standardization makes it approximate; main text says "expected value of LFS" |
| LFS-VC both-negative boundary branch not the constrained optimum | Yes in `rmfs/src/rmfs/components/variance.py`; all 104 saved variance-component records have zero negative components, so the branch never ran | Appendix gives the interior formulas, the boundary rule as implemented, and the zero-exposure statement; the 19 expansion profiles do not store the diagnostic (open item) |
| English-versus-multilingual test described more broadly than run | Yes: 12-donor panel, mean excludes English and target, PCA and standardization on the full grid, 3 folds, ridge 1, 64 PCs | Section 5 paragraph and conclusion ("most target languages") corrected |
| CVP means different things | Partly: both code paths measure within-language variance across sentences; the paper's "sentence-component" label was the error | Defined precisely in Section 7; "raw content variance" identified as the unstandardized version |
| Invariance tally attributed to the wrong measures | Yes: the two flags are 1e-8 tolerance on AaR and mean margin for Mistral-7B | Corrected |
| Statistical wording (exposure proxy, 0.315, sanity check, logit vs raw scale) | Yes | Corrected in abstract, setup, Section 6, and the regression appendix |
| "Dimensional collapse", "only raw content variance detected", VICReg, bounded penalty, "never used as an objective" | Yes | Reworded as magnitude contraction; norm and variance both detect; penalty scoped to the tested setup |
| Range and cohort wording (nine-fold cohort, 0.60 to 0.91, BLOOM, 0.3 to 8%, EuroLLM Arabic, offset plus scale, instruct caption, reproducibility statement, language supports) | Yes against the tables | Corrected |
| Subsampling intervals centered on a different estimate | Yes | Figure 1(b) redrawn: dot = resampling median, bar = interval, hollow marker = first-300 grid value; captions state the conditional-on-layer meaning |
| Citations | Yes | Added NLLB (FLORES-200), Raghu et al. 2017 (SVCCA), Glottolog; word-alignment loss cited in Section 7 |
| Mechanics | Yes | Models table column spec fixed; logit-lens aside moved to the measurement-dependence appendix to hold page 9 |

Still open: the final-copy switch stays on for Koehn's copy and must be commented out before upload; the anonymous repository link; the primary-model revisions (not in the local artifacts); the negative-component diagnostic for the 19 expansion-set profiles (cluster re-check on stored grids).
