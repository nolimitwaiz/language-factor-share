# Aim 1 Orthogonal-Basis Sensitivity Results

Date: 2026-08-24  
Status: complete  
Frozen protocol: `prereg/addenda/AIM1_AXIS_BASIS_SENSITIVITY_PROTOCOL_2026-08-24.md`  
CLSP array: `1761838` (14/14 tasks `COMPLETED`, all exit `0:0`)  
CLSP assembly: `1761870` (`COMPLETED`, exit `0:0`)

## 1. Support and completeness

The experiment used the frozen 14-model manifest, all 69 saved coarse layers, the fixed 13-language/300-concept grids, five frozen rotation seeds, and cumulative sweep counts 0, 1, 4, and 16. This produced 1,380 model-layer-seed-basis rows.

Every source `.npz` matched the source hash recorded by the earlier native-axis audit. The largest relative error in conserving either total `SS_language` or total `SS_concept` was `1.4697e-15`, far below the frozen `1e-10` abort threshold. Raw global direct-SS LFS was unchanged under every rotation; its model-layer range remained `0.3875` to `0.9901`, with median `0.7730`, at every sweep count.

## 2. Primary native-to-16-sweep result

The primary comparison averages the five frozen rotation seeds within model-layer and pairs all 69 layers between the native basis and sweep 16.

| Reading | Native median across layers | Sweep-16 median across layers | Median paired change | Paired IQR |
|---|---:|---:|---:|---:|
| Median coordinate language share | 0.874 | 0.836 | -0.014 | [-0.139, 0.011] |
| BY-significant coordinate fraction | 1.000 | 1.000 | 0.000 | [-0.00005, 0.000] |
| Gini of coordinate `SS_language` | 0.855 | 0.542 | -0.174 | [-0.257, -0.109] |
| Top-5%-coordinate share of total `SS_language` | 0.797 | 0.310 | -0.319 | [-0.428, -0.162] |

The BY-significant fraction remained essentially saturated. At sweep 16, the minimum layer fraction was `0.9966`; the median was `1.000`. The large language main effect therefore remains statistically widespread across these randomized orthogonal bases under the fixed grid and Gaussian/Beta reference.

Energy concentration is different. Every model had a negative median change in `SS_language` Gini; model-median changes ranged from `-0.257` to `-0.097`. Every model also had a negative median change in the top-5% language-energy fraction; changes ranged from `-0.462` to `-0.111`. Thus the strong concentration of absolute language energy in a small set of native coordinates is not basis invariant.

The median coordinate share itself was sensitive but did not move in one universal direction at every layer. Across model-level medians, its change was `-0.063`, with a range from `-0.175` to `0.018`. This is another reason not to interpret an individual hidden dimension as an identifiable scientific object.

## 3. Correct interpretation

The combined native and rotated evidence supports this statement:

> On the fixed balanced grids, systematic language main effects are widespread across both native and randomized orthogonal directions. Their apparent concentration in a small set of native coordinates is strongly basis dependent.

It does not support any of the following:

- that almost every hidden dimension is an independently causal language neuron;
- that the native high-energy axes are uniquely identifiable;
- that rotating the hidden state changes the model's represented information;
- that the Gaussian/Beta reference is a realistic generative model of trained hidden states;
- that coordinate-level significance predicts downstream task performance.

The axis analysis may appear in the paper only as a construct-validity/sensitivity result with this limitation stated directly. Any causal or mechanistic claim would require interventions or invariant subspace methods.

## 4. Relationship to the Profile result

This result reinforces the diagnostic-family framing. Global raw direct-SS LFS is exactly orthogonally invariant, while coordinate concentration is not. Profile R2-R4 describe other properties that LFS does not encode. A single scalar cannot preserve all of these distinctions.

The basis result does not repair RMFS v1 and does not create a new metric. RMFS v1 remains a negative scalar stress test; Profile 0.2 remains four separate readings.

## 5. Reproducibility record

Final assembled hashes:

| File | SHA-256 |
|---|---|
| `aggregate.json` | `2e9d77995209b146e8044d8518fe3e04afec15bd3cf95697d1925c8bf23d4840` |
| `all_basis_rows.csv` | `fadf5a1ccd2158cf54033ee3eb2d5b31494a76d41d42682f522c1d2764e062fb` |
| `manifest.json` | `215d879e314688c4bb0c5f48b5b3052873b8ae29b3afda71bbe74f963bc9ba5e` |
| `model_layer_sweep_seed_mean.csv` | `43acc4cecadfbf623790ccc84b5583abe8cfbf1aadc452a7a0245383a24fa073` |
| `model_median_deltas.csv` | `b3017a605ea50ee59e04cd8abc02cb3a367326ec58772b3a1a98fc1b3bd95d10` |
| `native_vs_sweep16_paired.csv` | `47fa45907b31de4a50bf83ec16adee2fbde13283ec024b4d88a8ea309e176c9e` |

All 28 per-model part files and manifests are retained under `results/round3/axis_basis_sensitivity_2026-08-24/parts/`.

## 6. Decision

Round 3 stage R0 is complete. No additional coordinate-axis computation is required for the current Aim 1 paper. The next scientific decision is R1: run the frozen 19-model content-transfer expansion, or remove the expanded content-transfer claim with Professor Koehn's agreement.
