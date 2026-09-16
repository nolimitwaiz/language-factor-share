"""Ledoit-Wolf covariance shrinkage (Phase 1.4).

Used wherever a covariance or consensus estimate is formed with dimension D
large relative to the sample count n — concept-mean covariances at D=2048
against a few hundred concepts sit squarely in the regime where the sample
covariance's eigenvalue spectrum is badly distorted (Marchenko-Pastur
spreading: top eigenvalues inflated, bottom deflated). Linear shrinkage
toward a scaled identity is the standard fix and is what the prior
campaign's quant framing would call covariance denoising.

Implementation wraps scikit-learn's LedoitWolf (a dependency we already
carry) rather than re-deriving the estimator; the value added here is the
typed result with diagnostics and the unit battery pinning the two
properties the pipeline relies on: intensity -> 0 as n/D grows, and
eigenvalue MSE beating the sample covariance on spiked models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

__all__ = ["ShrunkCovariance", "ledoit_wolf"]


@dataclass(frozen=True)
class ShrunkCovariance:
    """Shrunk covariance with its diagnostics; never a bare matrix."""

    covariance: np.ndarray      # (D, D)
    intensity: float            # shrinkage weight toward the identity target
    n: int
    D: int

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.sort(np.linalg.eigvalsh(self.covariance))


def ledoit_wolf(X: np.ndarray) -> ShrunkCovariance:
    """Linear Ledoit-Wolf shrinkage on rows-are-samples X of shape (n, D)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be (n, D); got shape {X.shape}")
    n, D = X.shape
    if n < 2:
        raise ValueError(f"need n >= 2 samples; got {n}")
    lw = LedoitWolf(assume_centered=False).fit(X)
    return ShrunkCovariance(covariance=lw.covariance_,
                            intensity=float(lw.shrinkage_), n=n, D=D)
