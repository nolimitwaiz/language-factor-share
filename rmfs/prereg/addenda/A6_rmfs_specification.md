# Addendum A6 — the RMFS specification (current; supersedes the v1 scalar
# aggregation for all forward-looking claims)

**Dated 2026-08-05, written AFTER the dev-grid exams (S1-S7, A5) and
BEFORE the blind walk-forward. The name of the metric is RMFS — no
version suffix in any user-facing artifact; provenance lives in this
addendum chain. The v1 soft-min scalar remains on the record as scored
(S1/S2 FAIL, S5/S6 PASS) and is not re-litigated.**

## Definition

RMFS is a per-language, three-channel instrument plus one flag:

- **Structure** q_L = 1 − LFS-VC (REML, z-track). Validated: deflated
  ρ = 0.76 vs held-out content transfer (S5 family).
- **Discrimination** = shared-prep MEXA at the frozen layer. Validated:
  deflated ρ = 0.24-0.31 vs downstream alpha (RW-corrected).
- **Damage tails** = shared-prep AaR (tail mean). Validated: ρ = +0.51
  on the fault-containing arm panel (only significant harm predictor).
- **Reorganization flag** = C_pres vs a frozen reference (intervention
  settings only). Reported ALWAYS; claims harm NEVER (A5: collapse was
  behaviorally benign on 0.6B/MCQ). Detection is exact ((1−m)², S2).

The per-language 3-vector (+ flag) is the metric of record. The scalar,
for ranking requests only:

    RMFS(model, ℓ) = mean of the three channels' PERCENTILES against
    the frozen 14-model development-grid distributions (per channel,
    per language where defined; linear interpolation, clipped to
    [0, 100]). Model-level RMFS = median over panel languages.

Percentile calibration is the S2 postmortem lesson (never fuse raw
heterogeneous scales) and is FROZEN from the dev grid as of this date;
walk-forward models are scored against these reference distributions,
never against themselves.

## Frozen walk-forward predictions (families untouched since prereg)

Models: Gemma-3-4B-pt (family A), Aya-Expanse-8B (family B), fallback
EXAONE-3.5-7.8B. Belebele 0-shot on the 41-task panel as the extrinsic.

- **W1 (computability):** all three channels + gates produce finite,
  non-degenerate values on both families; every channel value falls
  inside [dev-grid min − 20% range, max + 20% range].
- **W2 (validity):** pooled per-language deflated residual Spearman of
  scalar RMFS vs Belebele accuracy over the two families is positive
  with the 95% language-cluster CI excluding zero.
- **W3 (model-level ordering):** Aya-Expanse-8B (explicitly
  multilingual-tuned) scores higher model-level RMFS than Gemma-3-4B-pt.
- **W4 (channel coherence):** the discrimination channel's per-language
  ranking correlates ρ ≥ 0.5 with each family's own Belebele
  per-language ranking (raw, sanity-level).

Failures bind: any W that fails is reported as scored, no redesign.
Runs are blocked until the HF token is rotated and the Gemma license
accepted (user action; recorded open since 2026-08-02).
