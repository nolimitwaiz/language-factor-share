"""Single source of randomness for the repository (CLAUDE.md rule 11).

Every stochastic operation draws its generator from here, so a run's seed
appears in exactly one place in its manifest and the default seed set is
not re-declared ad hoc. The prior campaign's Aim-2 arms used seeds
{0, 1, 2}; re-analyses of those arms keep their original seeds (see
ledgers/INVENTORY.md) — DEFAULT_SEEDS applies to new work only.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEEDS: tuple[int, ...] = (13, 42, 71)


def rng(seed: int) -> np.random.Generator:
    """The only sanctioned way to obtain a Generator."""
    return np.random.default_rng(seed)


def spawn(seed: int, n: int) -> list[np.random.Generator]:
    """n independent child generators for parallel folds/workers.

    Uses SeedSequence spawning rather than seed arithmetic so that child
    streams are statistically independent and reproducible from the single
    parent seed recorded in the manifest.
    """
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]
