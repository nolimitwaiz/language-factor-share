# Estimator notes (running log, per the standing instruction)

Failure modes and subtleties met while building the measurement core. Paper
appendix material; every entry earned by an observed behavior, not theory.

## N1 (2026-08-02) — REML boundary pooling has a downward selection bias at tiny L

The nonnegative-REML closed form pools a component's sum of squares into the
error term when its ANOVA estimate is negative (boundary solution). Pooling
CONDITIONS on ms_L < ms_E, which selects low ss_L realizations into the
pooled error and biases sigma2_E slightly downward where boundaries are hit
often. Observed: pure noise at L=2, N=50, D=32 gives sigma2_E ≈ 30.3 vs the
true 32 (about 5%), with roughly half of dimensions on the L-boundary. At
L=12+ the effect is under 1%. Consequence: sigma2_E tolerances at small L
are 10% in the unit battery, and small-L configurations are flagged rather
than trusted for absolute error variances. The systematic components remain
correctly near zero under pure noise, which is what LFS-VC consumes.

## N2 (2026-08-02) — the legacy floor lives in the sum-of-squares scaling

The legacy share's pure-noise value (L−1)/((L−1)+(N−1)) exists BECAUSE the
language sum of squares is scaled by N and the concept sum by L (batch_lfs
convention). A first draft of the parity function omitted the multipliers
and produced a floor of (L−1)L/((L−1)L+(N−1)N) instead — numerically ~24x
smaller at (12, 300). The unit test on the known floor caught it before any
real data was touched. Lesson recorded because any future port of the
legacy estimator will face the same trap.

Addendum to N1: nonnegative truncation itself leaves ~0.4·SE of positive
mass per dimension in expectation (half-normal mean). "Components ≈ 0 under
pure noise" is therefore a df-scaled statement, not an absolute one: at
(L=2, N=50) the concept component's truncated mass is ~6% of sigma2_E and
that is CORRECT behavior, not a bug. The unit battery bounds each component
by 2.5 × 0.4 · D · SE(df) rather than a flat percentage.

## N3 (2026-08-02) — Ledoit-Wolf intensity semantics, and what it does not minimize

Two subtleties met while writing the shrinkage battery. (1) "Intensity -> 0
as n/D grows" presumes the true covariance DIFFERS from the shrinkage
target: for isotropic data the target equals the truth and full shrinkage
(intensity -> 1) is the optimal estimate — a first-draft test asserted decay
on isotropic data and failed correctly. Both behaviors are now asserted so
neither gets "fixed" later. (2) LW minimizes expected FROBENIUS risk of the
covariance, not the sorted-eigenvalue MSE; with strong spikes at n ~ D the
sample covariance can win on the eigenvalue metric while losing on
Frobenius (measured: 10.61 vs 11.91 at n=80, D=64). The battery pins the
Frobenius claim only.

Environment note: the local dev venv (python 3.9, numpy 2.0.2 on macOS
Accelerate) emits spurious divide/overflow RuntimeWarnings from BLAS on
trivially small matmuls while producing correct values (verified against
closed forms above). The authoritative unit-battery environment is the
cluster python; the Phase-1.7 grid sbatch runs `pytest tests/` as its
prologue so no grid number is produced on a machine where the battery has
not passed.

## N4 (2026-08-02) — M5 selection requires nonlinearity the kernel can exploit at the protocol n

Under the one-SE rule, the RBF-KRR rung is selected only when it beats the
ridge rung out-of-sample beyond split noise. At n~120 train concepts in a
12-dim latent, a saturating warp (Z + 0.4 tanh 2Z) leaves KRR underfit
(held-out SSE 42 vs ridge 17) and selection correctly stays at M2; an
un-saturating cubic warp is exploitable and M5 wins decisively (106 vs
199). Consequences: (1) this conservatism matches the real-data finding
that M5's held-out share is negative (purged audit: more negative still);
(2) battery calibration for the monotone-warp signature cell must verify
the injected magnitude is KRR-exploitable at the protocol n, otherwise the
cell measures estimator power rather than metric blindness.

## N5 (2026-08-02) — raw-track effective rank is dominated by massive activations

First grid numbers: effective rank of the RAW concept-mean covariance is
~1.0-2.0 for twelve of fourteen models at their dip layers — one or two
massive-activation dimensions own the raw spectrum — while the OLMo-2
family (known to lack massive activations) reads 24-78. As an observational
capacity gate, raw-track r_eff therefore measures rogue-dimension anisotropy
rather than concept capacity. Consequences: addendum A1 must specify the
TRACK for its r_eff threshold (the z-scored native track is the meaningful
one for capacity); the raw-track number is retained as a massive-activation
diagnostic in its own right. Cross-check queued in the next cheap pass.
Comparison against the prior grid at identical layers: legacy-estimator
values agree within 0.003-0.064 across the n=300 to n=1500 protocol change
(largest: salamandra-2b), so the construct is stable across stacks; my
initial alarm rested on misremembered numbers, and the committed tables
settled it.

## N6 (2026-08-03) — reverse-direction kernel maps lose their forward advantage

Fitting language states FROM the consensus (reconstruction direction), RBF
kernel ridge underperforms plain ridge on the same cubic worlds where the
forward fit wins decisively (reverse: KRR MSE 0.36 vs ridge 0.29 at d=12,
n=200, stable across amplitude and train size; forward: KRR 106 vs ridge
199 SSE in the ladder battery). The asymmetry matters because rung
selection is frozen from the forward direction: languages selected at M5
get reverse-KRR reconstructions that may be weaker than a reverse-ridge
would have been. A2 (amended before freeze) discloses this and marks all
M5-rung T cells; the affected population is exactly the OLMo-2 and Falcon3
families, which are also the massive-activation-free anomaly — the two
observations may not be independent.
