"""Incident #1 encoded as tests: mean-pool in fp32 ALWAYS.
Massive-activation outliers overflow fp16 sums — this happened."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from src.campaign.dump_embeddings import pool_hidden_states  # noqa: E402


def _fake_batch(B=2, T=32, D=8, magnitude=1.0, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    hs = [torch.randn(B, T, D, generator=g).to(dtype) * magnitude
          for _ in range(3)]
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, T // 2:] = 0                      # padding on the first sequence
    return hs, mask


def test_pool_matches_pilot_formula():
    """pool_hidden_states must equal the verbatim pilot_metrics pooling."""
    hs, mask = _fake_batch()
    got = pool_hidden_states(hs, mask, [0, 2])
    m = mask.unsqueeze(-1).float()
    ref = [(h.float() * m).sum(1) / m.sum(1) for h in hs]  # pilot lines
    ref = torch.stack(ref, 1).float().numpy()[:, [0, 2], :]
    np.testing.assert_allclose(got, ref, rtol=0, atol=0)


def test_fp16_pooling_canary():
    """A test that FAILS if anyone pools in fp16: with massive activations,
    the fp16 token-sum overflows to inf while the fp32 path stays finite and
    correct. Same-sign large values guarantee the sum exceeds fp16 max
    (65504) regardless of the accumulator's internal precision — the result
    still overflows on the fp16 output cast."""
    B, T, D = 2, 64, 8
    g = torch.Generator().manual_seed(0)
    hs = [(torch.rand(B, T, D, generator=g) * 20000 + 10000).half()
          for _ in range(2)]                        # values in [1e4, 3e4]
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, T // 2:] = 0
    # our path: fp32 pooling of fp16 hidden states — finite, correct
    got = pool_hidden_states(hs, mask, [0])
    assert np.isfinite(got).all()
    ref64 = (hs[0].double() * mask.unsqueeze(-1).double()).sum(1) \
        / mask.unsqueeze(-1).double().sum(1)
    np.testing.assert_allclose(got[:, 0, :], ref64.numpy(), rtol=1e-3)
    # the forbidden path: fp16 accumulation overflows (this is the incident)
    m16 = mask.unsqueeze(-1).half()
    bad = (hs[0] * m16).sum(1) / m16.sum(1)
    assert not torch.isfinite(bad).all(), \
        'fp16 pooling no longer overflows — canary invalid, re-examine'
