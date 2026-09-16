# Aim 1 Addendum: Orthogonal-Basis Sensitivity of Coordinate Language Shares

Frozen: 2026-08-24, before inspecting any rotated-basis outcome  
Scope: Aim 1 construct validity only  
Status at freeze: native-axis results opened; rotated-basis results not computed

## 1. Question

The native-coordinate audit found that language shares are statistically widespread while absolute language-effect energy is concentrated. Hidden-state coordinates are not uniquely identifiable: an orthogonal rotation changes individual axes without changing the represented Euclidean geometry. This addendum asks which coordinate-level conclusions survive exact orthogonal changes of basis.

## 2. Fixed data and support

- The same 14-model manifest used by the completed Round 3 anisotropy audit.
- The same 69 saved coarse layers.
- The same fixed 13-language, first-300-concept balanced grids.
- No new representation extraction and no model forward pass.
- Source `.npz` hashes must match the hashes recorded by the completed anisotropy manifests.

## 3. Exact transformation

For each model-layer grid, compute the language-effect matrix and concept-effect matrix after grand-mean removal. Apply the same sequence of random Givens rotations to their coordinate columns. One sweep randomly permutes all coordinates and rotates each adjacent pair by an independently sampled angle uniformly distributed on `[0, 2*pi)`. Every sweep is an exact orthogonal change of basis.

The cumulative sweep counts are fixed at:

- 0: native basis;
- 1: one full random-pair sweep;
- 4: four cumulative sweeps;
- 16: sixteen cumulative sweeps.

The five fixed random seeds are `1729`, `2718`, `3141`, `5772`, and `8111`. Seed 0 is not added after inspection.

## 4. Fixed readings

At every model, layer, seed, and sweep count, report:

1. median coordinate language share;
2. coordinate-share quantiles 0.05, 0.25, 0.50, 0.75, and 0.95;
3. fractions of coordinates with language share above 0.50, 0.75, and 0.90;
4. fraction declared language-heavy using the same one-sided Gaussian/Beta reference and Benjamini-Yekutieli correction as the native audit;
5. Gini coefficient of coordinate `SS_language`;
6. fraction of total `SS_language` in the top 5% of coordinates ranked by `SS_language`;
7. raw global direct-SS LFS from the total language and concept sums of squares;
8. relative conservation errors for total `SS_language` and `SS_concept`.

No aggregate Profile or RMFS score will be constructed.

## 5. Primary comparison

The primary contrast is paired within model-layer between the native basis and sweep 16. Summarize the distribution of changes across all 69 model-layers by median, interquartile range, minimum, and maximum. Report model-level medians as a sensitivity check so architectures with five saved layers do not silently dominate architectures with four.

The primary quantities are change in:

- median coordinate language share;
- BY-significant coordinate fraction;
- `SS_language` Gini;
- top-5%-`SS_language` energy fraction.

Intermediate sweeps show the trajectory but are secondary.

## 6. Predictions and interpretation

1. Raw global direct-SS LFS and total factor energies must be conserved to numerical tolerance. Failure above `1e-10` relative error aborts the result.
2. Coordinate-level concentration readings are expected to change under mixing. A substantial change confirms basis dependence.
3. If the BY-significant fraction remains near one after rotation, the correct conclusion is that a large language main effect is widespread across random directions under this fixed grid, not that the model contains thousands of identifiable language neurons.
4. If the Gini and top-5% energy fraction fall after rotation, native-axis energy concentration is a coordinate-system property and cannot be treated as an invariant subspace result.

## 7. Statistics and multiplicity

This is a deterministic sensitivity analysis over five frozen random bases, not a population-inference study. Report paired descriptive distributions and all five seeds. Do not attach confirmatory p-values. The BY correction is retained only to reproduce the original axis-classification rule within each basis.

## 8. Abort and reporting rules

- Abort a model if any expected layer or fixed language is absent.
- Abort the aggregate if any model output or seed/sweep cell is absent.
- Abort if the exact orthogonal invariants exceed `1e-10` relative error.
- Report all directions, including a null sensitivity result.
- Do not make a causal-neuron or identifiable-axis claim under any outcome.
- Do not edit the technical report until the output audit and result memorandum are complete.
