# Addendum A2 — reconstruction specification for behavioral T

**Dated 2026-08-03, written BEFORE any T extraction or computation, per the
frozen prereg's addendum rule and the kickoff's explicit stop-and-write
instruction for design forks in 2.1. No T value exists at the time of
writing; the geometric components' deflated nulls are known (Phase 1
report), which is disclosed because it means T now carries the validity
case — a reason for MORE specification discipline here, not less.**

## The two forks the prereg does not settle

The prereg (§3.1) defines B_recon as the NLL reduction when the context
sentence's token states are "reconstructed from the shared factor through
ℓ's selected ladder map (fit on training concepts, applied to held-out
concepts, document-purged)." Two implementation choices are underdetermined:

**Fork 1 — map direction.** The ladder maps and the rung selection behind K
are fit ℓ → consensus. Reconstruction needs consensus → ℓ. The selected
rungs on the real grid are overwhelmingly M4 (ridge) and, for two families,
M5 (kernel); neither is invertible, so "apply the inverse" is not available.

**Frozen resolution:** reconstruction uses the SELECTED RUNG (per language,
from the Phase-1 K selection — the selection itself is not refit) with a map
of that rung's family fit in the REVERSE direction, consensus → ℓ, on the
same document-purged training concepts, with the same hyperparameter
procedure (inner CV for ridge lambda and kernel bandwidth/lambda). Rungs
M0–M3 are exactly invertible, and for them the reverse fit coincides with
the inverse up to estimation noise; M4/M5 reverse fits are proper
regressions of ℓ's coordinates on consensus coordinates. This preserves
the construct — "what the shared factor predicts about ℓ" — for every rung
without ad-hoc pseudo-inverses.

**Fork 2 — the out-of-subspace component.** The maps live in the rank-r
pooled subspace (the published rank per model). Token states have large
components outside it — including the massive-activation dimensions that
dominate raw spectra (estimator note N5). "Predicted from the shared
factor" cannot include token-specific content the factor never modeled.

**Frozen resolution (primary):** the reconstructed token state is

    x̂_t = μ_raw + B · m_ℓ( Bᵀ (x_t − μ_raw) )

where B is the rank-r basis, μ_raw the pooled mean (which carries the
static part of the massive-activation dimensions), and m_ℓ the reverse
rung map. Everything token-specific outside the subspace is REPLACED by
the pooled mean: the reconstruction carries exactly the factor-predicted
content plus the global offset, nothing else. This is the strict reading
of the construct.

**Disclosed sensitivity (reported, never selected on):** the hybrid
variant x̂_t = x_t + B·(m_ℓ(Bᵀ(x_t−μ_raw)) − Bᵀ(x_t−μ_raw)) keeps the
original out-of-subspace content and remaps only the in-subspace part. It
leaks ℓ-specific information by construction (disclosed weakness) but
guards against the failure mode where the strict variant destroys enough
of the state that all NLLs saturate and T is uniformly ≈ 0 for reasons of
state destruction rather than factor content. If the strict primary
saturates (defined below), the paper reports both and says so plainly.

**Saturation flag (frozen now):** the strict primary is declared saturated
for a (model, language) cell if B_recon < B_mismatch − ε_B, i.e.
reconstruction is WORSE than a wrong-document context by more than the
degenerate-denominator threshold. Saturated cells are reported as
saturated, not folded into means.

## Disclosed limitation, measured before any extraction

The reverse direction does not inherit the forward direction's kernel
advantage. On synthetic cubic worlds at the battery's n and d, reverse-KRR
(consensus → ℓ) UNDERPERFORMS reverse-ridge (MSE 0.36 vs 0.29 at d=12,
n=200; the gap persists across amplitudes and train sizes), while
forward-KRR wins its own world decisively in the ladder battery. Estimator
note N6 records the observation. Consequence, stated now: for languages
whose SELECTED rung is M5 (all 128 languages of both OLMo-2 models and 120
of Falcon3's), B_recon carries reverse-map weakness that is a property of
the reconstruction, not of the representation. Those cells' T values are
therefore reported with a `rung=M5` marker, and any cross-model comparison
involving the all-M5 families says so in the caption. The battery
requirement on reverse maps is correspondingly what the spec actually
needs: each family fits finitely, deterministically under seed, and beats
the mean predictor on its own world — NOT that reverse-KRR beats
reverse-ridge.

## What is NOT changed by this addendum

T's formula, ε_B = 0.072, the clipping rule, the document-purged
concept-disjoint requirement, the per-token application of sentence-level
maps (offset/scale trivially per token; M3 rotation per token; M4/M5 per
token on subspace coordinates), and the harness (3.1.3 pair set, wrong-
document mismatch control) — all as frozen in the prereg and kickoff.

## Order of operations this addendum binds

1. Unit battery for transfer.py on synthetic worlds with known answers
   (identity/offset worlds → T ≈ 1; pairing permutation → T ≈ 0; ε_B rule;
   clipping; purge enforcement) — before any real extraction.
2. Extraction on pretrained models and arms only after the battery is
   green on the cluster python.
