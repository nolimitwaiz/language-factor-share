# Pilot Report — LFM+AaR metric suite v0

**Date:** 2026-07-04 · **Model:** Qwen3-0.6B-Base (28 layers + emb, d=1024, fp16/MPS)
**Data:** NTREX-128, 100 parallel sentences, 26 languages + English pivot
(10 high / 8 mid / 8 low resource) · **Runtime:** ~3 min on M5 Pro (24 GB), local
**Code:** `code/pilot_metrics.py` · **Outputs:** `metrics.json`, `fig_pilot_overview.png`

## Findings

### 1. The LFS layer-curve is U-shaped — the three-stage hypothesis, measured
Language-Factor Share (M1: fraction of concept+language variance owned by language
identity) starts at **0.91–0.94** (embedding/early layers), dips to **0.60–0.64**
across layers 8–16, returns to **0.88–0.91** at the output layers. MEXA is the
mirror image (peak 0.632 at layer 18). This is the NSF proposal's §3.1.1
"language-specific → shared → language-specific" processing story, produced as a
single measured curve by variance decomposition — not asserted from prior work.

### 2. The mean hides the tail — first evidence for Alignment-at-Risk (M3)
At the most-shared layer (18), by resource tier:

| tier | MEXA | mean margin | AaR@10 (worst decile) |
|---|---|---|---|
| high (n=10) | 0.940 | +0.196 | +0.035 |
| mid (n=8) | 0.775 | +0.126 | **−0.000** |
| low (n=8) | 0.103 | −0.056 | −0.149 |

The key row is **mid**: MEXA = 0.775 looks healthy, but the worst decile of
sentences is already at the retrieval-failure boundary. Mean-level metrics
(MEXA, IP) cannot see this. This is the AaR pitch in one table.

### 3. Hub test (M2): all 26/26 languages are better explained by the latent
factor than by English (mean R² advantage +1.04), and the advantage is *largest
for low-resource languages* (khm +1.75, ben +1.66, zul +1.61) and smallest for
Romance/Germanic (por +0.62, spa +0.72). Directionally consistent with
Shani & Basirat 2025 and Bafna et al. 2025 (shared space ≠ English), against
naive "thinks in English."

## Caveats (fix before believing finding 3)

- **Hub-test R² values are all negative** (CV with N=100, 64 PCs) — the
  *comparison* is like-for-like but absolute fit is poor. Needs N≥500 sentences.
- **Smoothness confound:** the latent factor is a 24-language mean → lower
  variance → easier ridge target. Control needed: variance-matched latent, and
  single-language baselines (is deu→fra also better than eng→fra?).
- One model, one size, NTREX only. Scaling curve (0.6B → 4B → 8B) and FLORES
  cross-check are the immediate next runs — all local.
- MEXA here is our reimplementation (strict row+column dominance, weighted-mean
  pooling ≈ their `embd_weighted`); validate against their published scores
  before any comparison claim.

## Next runs (all Mac-local)

1. `--n_sents 500` re-run (hub-test stability)
2. Qwen3-4B + Llama-3.2-3B or Qwen2.5-7B → does the LFS dip *deepen* with scale
   and multilinguality? (hypothesis: yes, and correlates with downstream)
3. Variance-matched hub control + single-language baselines
4. Deflation prep: pull published MEXA scores + Belebele per-language accuracies
   + per-language token counts (OLMo-2/BLOOM exact)
