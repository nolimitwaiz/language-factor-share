"""The LFS estimator's null behaviour.

Under a pure-noise null, the two-way decomposition returns the
degrees-of-freedom ratio exactly:

    LFS_null = (L - 1) / ((L - 1) + (N - 1))

for L languages and N concepts, independent of dimensionality and of the
data. Derivation: with X[l, c, :] iid, the concept means carry (N-1) degrees
of freedom at variance s^2 / L and the language means carry (L-1) degrees of
freedom at variance s^2 / N; the leading L and N factors in the sums of
squares cancel those, leaving E[ss_lang] = (L-1) D s^2 and
E[ss_concept] = (N-1) D s^2.

This is the pure-noise value, NOT a lower bound. Writing the grid as
X[l, c] = C[c] + G[l] + E[l, c] and expanding,

    ss_lang    = (L-1) D s^2 + N * SS_G
    ss_concept = (N-1) D s^2 + L * SS_C

so the numerator carries a noise term (L-1) D s^2 that does not vanish. A
grid with strong concept structure and weak language structure therefore
measures *below* the pure-noise value, which the tests here confirm.

Consequences, stated in the report at section 5.3:
  * A small measured LFS is not automatically evidence of
    language-agnosticism: the numerator's noise term must be small relative
    to N * SS_G for the value to be read as language structure.
  * Differences at matched (L, N) are unaffected, because the noise term is
    common to both. Dip depth, cross-model comparison at fixed grid, and
    before/after injection contrasts are therefore unaffected.
  * Absolute LFS values are not comparable across different (L, N): the
    pure-noise value moves from 0.298 at (128, 300) to 0.078 at (128, 1500).
"""
import numpy as np
import pytest

from pilot_metrics import lfs


def df_floor(L, N):
    return (L - 1) / ((L - 1) + (N - 1))


@pytest.mark.parametrize("L,N,D", [
    (2, 700, 256), (12, 120, 256), (5, 50, 128),
    (128, 300, 64), (128, 1500, 64), (20, 200, 128),
])
def test_null_equals_df_floor(L, N, D):
    """Pure noise returns the degrees-of-freedom ratio, not zero."""
    rng = np.random.default_rng(L * 1000 + N)
    vals = [lfs(rng.standard_normal((L, N, D)))['lfs'] for _ in range(8)]
    assert abs(float(np.mean(vals)) - df_floor(L, N)) < 0.005


def test_floor_is_dimension_independent():
    """The floor depends on the grid shape, never on the representation
    dimension."""
    rng = np.random.default_rng(0)
    got = [float(np.mean([lfs(rng.standard_normal((8, 200, D)))['lfs']
                          for _ in range(6)])) for D in (32, 256, 1024)]
    assert max(got) - min(got) < 0.005
    assert all(abs(g - df_floor(8, 200)) < 0.005 for g in got)


def test_pure_noise_value_is_not_a_lower_bound():
    """Strong concept structure with weak language structure measures BELOW
    the pure-noise value, because the concept sum of squares grows while the
    numerator's noise term does not. Recorded so that the pure-noise value
    is never described as a floor."""
    rng = np.random.default_rng(1)
    L, N, D = 10, 300, 128
    concepts = rng.standard_normal((N, D)) * 3.0
    X = np.stack([concepts + rng.standard_normal(D) * 0.5
                  + 0.3 * rng.standard_normal((N, D)) for _ in range(L)])
    assert lfs(X)['lfs'] < df_floor(L, N)


def test_noise_term_cancels_in_differences():
    """Two conditions measured at the same (L, N) can be compared directly:
    the numerator's noise term is common to both, so an ordering in measured
    LFS reflects an ordering in language structure. This is why dip depth
    and cross-model contrasts at a fixed grid are unaffected."""
    rng = np.random.default_rng(1)
    L, N, D = 10, 300, 128
    concepts = rng.standard_normal((N, D)) * 3.0

    def build(lang_strength):
        X = np.stack([concepts + rng.standard_normal(D) * lang_strength
                      + 0.3 * rng.standard_normal((N, D)) for _ in range(L)])
        return lfs(X)['lfs']

    assert build(3.0) > build(1.0) > build(0.5)


def test_real_signal_far_exceeds_floor():
    """A grid with genuine language structure must sit well above the floor,
    which is what separates a measurement from an artifact. Mirrors the
    monolingual-null experiment run on saved representations."""
    rng = np.random.default_rng(2)
    L, N, D = 12, 120, 256
    concepts = rng.standard_normal((N, D)) * 2.0
    X = np.stack([concepts + rng.standard_normal(D) * 2.0
                  + 0.3 * rng.standard_normal((N, D)) for _ in range(L)])
    assert lfs(X)['lfs'] > df_floor(L, N) + 0.2
