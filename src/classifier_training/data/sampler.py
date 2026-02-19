"""Tracking-aware weighted random sampler for monitoring class balance."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import WeightedRandomSampler


class TrackingWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler that records the last set of sampled indices.

    Used by SamplerDistributionCallback to visualise which classes were
    actually drawn in each epoch without modifying the sampling logic.

    The attribute ``_last_indices`` is populated after each full iteration
    (i.e. after the DataLoader has consumed all samples for one epoch).
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(
            weights=weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )
        self._last_indices: list[int] = []

    def __iter__(self) -> Iterator[int]:
        """Yield sampled indices and record them in ``_last_indices``."""
        indices = list(super().__iter__())
        self._last_indices = indices
        yield from indices
