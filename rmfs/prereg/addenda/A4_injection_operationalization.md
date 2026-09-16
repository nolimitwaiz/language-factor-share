# Addendum A4 — factor-state injection operationalization for B_recon

**Dated 2026-08-03, written after the strict/hybrid saturation was measured
and the harness was validated exactly (identity patches reproduce native
NLL to +0.0000; RUNS 1705946), and BEFORE any T array. This is the single
disclosed redesign; if injection also saturates, the frozen downgrade path
activates with both attempts on record.**

## Why the frozen operationalization failed, in one paragraph

Substituting a broadcast sentence-level factor state for real context token
states at the dip layer destroys more usable signal than it carries: the
reconstruction recovered ~10–20% of the mean-patch-to-native gap and less
than wrong-document TEXT. The failure is asymmetric by construction — the
native condition benefits from token-level content the factor never
modeled, so B_recon competes against a condition it cannot reach. The
construct ("what the shared factor predicts") was being measured THROUGH a
destruction penalty.

## The redesign: injection instead of substitution

No real context text in the reconstruction conditions. The context slot is
filled with NEUTRAL SCAFFOLD tokens (the separator token, repeated to match
the native context's token count — no new free parameter), and the dip-layer
states of those positions are replaced with the factor prediction:

    condition scaffold      : scaffold tokens, unpatched          → NLL_scaf
    condition inject        : scaffold, states ← μ + B·m_ℓ(z_i)   → NLL_inj
    condition inject-wrong  : scaffold, states ← μ + B·m_ℓ(z_j)   → NLL_injw
                              (z_j = the wrong-document partner's factor)

    T_inj = (NLL_injw − NLL_inj) / (NLL_mismatch − NLL_native)

Numerator: the CONCEPT-SPECIFIC benefit carried by the injected factor
state (right concept vs wrong concept, identical scaffold, identical
geometry — the only difference is which concept's factor is injected).
Denominator: the concept-specific benefit carried by real text (native vs
wrong-document), exactly as frozen in the prereg. T_inj is therefore "the
fraction of the text-borne concept-specific benefit recoverable from the
shared factor alone." ε_B = 0.072 applies to the denominator unchanged;
the saturation analog is NLL_inj > NLL_injw + ε_B (the right concept's
factor HURTS relative to the wrong one's).

Properties: no leak (no ℓ text anywhere in the numerator conditions); no
destruction asymmetry (both numerator conditions carry identical scaffold
damage, which cancels); the prereg's words — "states predicted from the
shared factor" as the context — are satisfied more literally than the
substitution ever did, since the factor states are now the ONLY
concept-bearing context.

## What is retained and what is recorded

- The strict/hybrid saturation results from the smoke stand as the record
  of the first operationalization; they appear in the paper's appendix
  with the identity-patch validation.
- Reverse maps, LOO consensus, rung inheritance, ε_B, purged pairs, and
  the harness are unchanged (A2/A3).
- The battery-order rule holds: injection is smoked (same 4-language,
  8-pair configuration) and must produce a majority of non-saturated,
  defined cells before any array.
