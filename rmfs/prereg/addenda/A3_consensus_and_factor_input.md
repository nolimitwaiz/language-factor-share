# Addendum A3 — consensus construction and reconstruction input for T

**Dated 2026-08-03, written BEFORE any T extraction, superseding one
formula in A2 after a design-review catch. No T value exists.**

## The catch: A2's fork-2 formula had a frame mismatch that would inflate T

A2 wrote the strict reconstruction as x̂_t = μ + B·m_ℓ(Bᵀ(x_t − μ)) — the
reverse map applied to the TOKEN'S OWN subspace coordinates. Those
coordinates live in language ℓ's frame; m_ℓ expects CONSENSUS-frame input.
Worse than a frame error, it is a leak: the reconstruction would carry
ℓ's own token content nearly unchanged (for small maps, m_ℓ ≈ identity on
its input), making B_recon ≈ B_native and T ≈ 1 regardless of what the
shared factor actually predicts. Caught while designing the extraction,
before any run; recorded here rather than silently rewritten because A2
was already committed.

## Frozen resolutions

**1. Reconstruction input is the sentence-level shared factor, not token
coordinates.** For concept i and language ℓ:

    z_i = leave-one-out consensus coordinates of concept i
          (train-fit GPA transforms applied to the OTHER languages'
          held-out sentence embeddings; exclude = ℓ)
    x̂   = μ_raw + B · m_ℓ(z_i)

- **Strict (primary):** every context token position receives x̂ — the
  factor-predicted sentence-level state, broadcast. Only factor content
  plus the pooled mean survives; token/positional variation from ℓ is
  gone, which is the honest price of the strict construct. "Per-token
  application" in the kickoff is satisfied in the only non-leaking form
  available without cross-language token alignment.
- **Hybrid (disclosed sensitivity, reported never selected on):**
  x̂_t = x_t − B·Bᵀ(x_t − μ_raw) + B·m_ℓ(z_i): the token's out-of-subspace
  content (positional scaffolding, massive-activation structure) is kept,
  the in-subspace content is replaced by the factor prediction. Leaks
  out-of-subspace ℓ content by construction; that is its stated role.

**2. Leave-one-out consensus is the PRIMARY for T** (exclude = ℓ, the
machinery legacy `consensus_target(..., exclude=)` already provides). The
self-inclusive consensus (1/L leakage, the K-ladder's disclosed
convention) is the sensitivity column for T, not the primary: K measures
map complexity, where 1/L self-inclusion is a small bias; T measures
recoverable content, where self-inclusion is a direct leak of the answer.

**3. Arms inherit the base model's per-language rung selection** (Phase-1
K on Qwen3-0.6B-Base), frozen — an arm's fine-tuning does not get to
re-select its own map family, or arms would be scored by different rulers.
Basis, μ, GPA and reverse maps ARE refit per arm on the arm's own
embeddings (the ruler's calibration follows the representation; the
ruler's FORM does not).

**4. Models without a Phase-1 rung selection** (Llama-3.1-8B, granite,
Yi — no campaign dumps) run in-job selection at the n=300 protocol before
their T, as their own labeled task; they are not blocked on, and do not
block, the dump-covered models.

A2's remaining content (reverse-direction fits, saturation flag, N6
disclosure, ε_B) is unchanged.
