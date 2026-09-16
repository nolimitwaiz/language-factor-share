"""Battery worlds + scoring: closed forms, conventions, legacy wiring."""

import numpy as np
import pytest

from rmfs.battery.score import (
    CellVerdict,
    floors_from_identity,
    score_cell,
)
from rmfs.battery.worlds import (
    AFFECTED_LANG,
    CONFIGS,
    LEGACY,
    affected_languages,
    apply_config,
)
from rmfs.components.variance import variance_components


def _world(seed=13, L=5, N=60, D=32):
    rng = np.random.default_rng(seed)
    concepts = rng.standard_normal((N, D)).astype(np.float32)
    E = {}
    langs = ["eng", "deu", "fra", "spa", "por"][:L]
    for lg in langs:
        off = rng.standard_normal(D).astype(np.float32) * 0.5
        noise = 0.3 * rng.standard_normal((N, D)).astype(np.float32)
        E[lg] = concepts + off + noise
    return E


def _basis(E, r=8):
    flat = np.concatenate(list(E.values()), 0).astype(np.float64)
    mean = flat.mean(0, keepdims=True)
    _, vecs = np.linalg.eigh((flat - mean).T @ (flat - mean))
    return vecs[:, ::-1][:, :r], mean


def test_configs_cover_all_csv_rows():
    names = {(n, m) for n, m in CONFIGS}
    assert len(names) == 15                       # 12 families, 15 configs
    assert ("uniform_collapse", "0.5") in names
    assert ("rotation_per_language", "0.8") in names


def test_uniform_collapse_cpres_closed_form():
    E = _world()
    b, mu = _basis(E)
    for m in (0.9, 0.5, 0.1):
        out, _ = apply_config(E, "uniform_collapse", str(m), 13, b, mu)
        X0 = np.stack(list(E.values()), 0)
        X1 = np.stack([out[lg] for lg in E], 0)
        vc0, vc1 = variance_components(X0), variance_components(X1)
        c_pres = vc1.sigma2_C_reml / vc0.sigma2_C_reml
        assert c_pres == pytest.approx((1 - m) ** 2, abs=1e-3)


def test_uniform_collapse_lfs_blind():
    # the founding result: global collapse leaves the REML ratio unchanged
    E = _world()
    b, mu = _basis(E)
    out, _ = apply_config(E, "uniform_collapse", "0.5", 13, b, mu)
    v0 = variance_components(np.stack(list(E.values()), 0)).lfs_vc
    v1 = variance_components(np.stack([out[lg] for lg in E], 0)).lfs_vc
    assert v1 == pytest.approx(v0, abs=1e-6)


def test_collapse_per_language_only_touches_declared_lang():
    E = _world()
    b, mu = _basis(E)
    out, _ = apply_config(E, "collapse_per_language", "0.5", 13, b, mu)
    for lg in E:
        if lg == AFFECTED_LANG:
            assert not np.allclose(out[lg], E[lg])
            v0 = ((E[lg] - E[lg].mean(0)) ** 2).mean()
            v1 = ((out[lg] - out[lg].mean(0)) ** 2).mean()
            assert v1 / v0 == pytest.approx(0.25, abs=1e-6)
        else:
            np.testing.assert_array_equal(out[lg], E[lg])


def test_identity_is_noop_and_permute_reorders():
    E = _world()
    b, mu = _basis(E)
    out, _ = apply_config(E, "identity", "-", 42, b, mu)
    for lg in E:
        np.testing.assert_array_equal(out[lg], E[lg])
    out, meta = apply_config(E, "pairing_permutation", "-", 13, b, mu)
    np.testing.assert_array_equal(out["eng"], E["eng"])   # pivot untouched
    assert not np.allclose(out["deu"], E["deu"])
    assert sorted(meta["perms"]["deu"]) == list(range(60))


def test_rotation_seeded_and_deterministic():
    E = _world()
    b, mu = _basis(E)
    a1, _ = apply_config(E, "rotation_per_language", "0.4", 13, b, mu)
    a2, _ = apply_config(E, "rotation_per_language", "0.4", 13, b, mu)
    np.testing.assert_array_equal(a1["deu"], a2["deu"])
    a3, _ = apply_config(E, "rotation_per_language", "0.4", 42, b, mu)
    assert not np.allclose(a1["deu"], a3["deu"])


def test_affected_languages():
    langs = ["eng", "deu", "fra"]
    assert affected_languages("offset_per_language", langs) == ["deu", "fra"]
    assert affected_languages("collapse_per_language", langs) == ["deu"]
    assert LEGACY.PIVOT == "eng"


# ------------------------------------------------------------- scoring
def test_floors_and_band():
    reps = [{"q_L": 0.500}, {"q_L": 0.502}, {"q_L": 0.498}]
    fl = floors_from_identity(reps)
    v = score_cell("identity", "-", "q_L", "band", 0, 3,
                   0.5, 0.503, fl["q_L"])
    assert v.passed                     # 0.003 <= 3*0.002
    v = score_cell("identity", "-", "q_L", "band", 0, 3,
                   0.5, 0.52, fl["q_L"])
    assert not v.passed


def test_down_up_ceiling():
    assert score_cell("x", "-", "q_L", "down", 0, 3, 0.5, 0.4, 0.01).passed
    assert not score_cell("x", "-", "q_L", "down", 0, 3, 0.5, 0.49, 0.01).passed
    assert score_cell("x", "-", "q_L", "up", 0, 3, 0.5, 0.6, 0.01).passed
    assert score_cell("x", "-", "RMFS", "ceiling", 0, 3, 0.5, 0.1, 0.01).passed
    assert not score_cell("x", "-", "RMFS", "ceiling", 0, 3, 0.5, 0.54,
                          0.01).passed


def test_down_mag_requires_monotonicity():
    fam = [0.30, 0.20, 0.10]            # increasing severity, decreasing T
    v = score_cell("uniform_collapse", "0.9", "T", "down_mag", None, 3,
                   0.5, 0.30, 0.01, family_values=fam)
    assert v.passed
    fam_bad = [0.30, 0.45, 0.10]
    v = score_cell("uniform_collapse", "0.9", "T", "down_mag", None, 3,
                   0.5, 0.30, 0.01, family_values=fam_bad)
    assert not v.passed


def test_exact_and_rungs():
    v = score_cell("uniform_collapse", "0.5", "C_pres", "exact", 0.25, 3,
                   1.0, 0.256, 0.001)
    assert v.passed and v.threshold == 0.01
    assert not score_cell("uniform_collapse", "0.5", "C_pres", "exact", 0.25,
                          3, 1.0, 0.262, 0.001).passed
    assert score_cell("r", "0.4", "q_K", "rung_ge", "M3", 3, 0, "M4",
                      0.0).passed
    assert not score_cell("r", "0.4", "q_K", "rung_ge", "M3", 3, 0, "M2",
                          0.0).passed
    assert score_cell("o", "-", "q_K", "rung_le", "M1", 3, 0, "M1",
                      0.0).passed


def test_verdict_is_frozen_dataclass():
    v = score_cell("x", "-", "T", "band", 0, 3, 0.1, 0.1, 0.01)
    assert isinstance(v, CellVerdict)
    with pytest.raises(AttributeError):
        v.passed = False
