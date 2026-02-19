"""Tracking-aware weighted random sampler for monitoring class balance."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal

import torch
from pydantic import BaseModel
from torch.utils.data import WeightedRandomSampler


class SamplerConfig(BaseModel, frozen=True):
    """Configuration for the training data sampler.

    Modes:
        disabled: No sampler — DataLoader uses shuffle=True (default).
        auto: Inverse-frequency weighting computed from train class counts.
        manual: User-provided per-class weights from config.
    """

    mode: Literal["disabled", "auto", "manual"] = "disabled"
    class_weights: dict[str, float] | None = None
    num_samples: int | None = None
    replacement: bool = True


def build_sampler(
    config: SamplerConfig,
    labels: Sequence[int],
    class_to_idx: dict[str, int],
) -> TrackingWeightedRandomSampler | None:
    """Factory: build a sampler from config and dataset labels.

    Args:
        config: SamplerConfig specifying mode and optional weights.
        labels: Per-sample class indices from the training dataset.
        class_to_idx: Mapping from class name to index.

    Returns:
        TrackingWeightedRandomSampler for auto/manual, None for disabled.
    """
    if config.mode == "disabled":
        return None

    num_classes = len(class_to_idx)
    num_samples = config.num_samples if config.num_samples is not None else len(labels)

    if config.mode == "auto":
        # Inverse-frequency weighting
        counts = torch.zeros(num_classes)
        for label in labels:
            counts[label] += 1.0
        class_weights = 1.0 / counts.clamp(min=1.0)
        class_weights = class_weights / class_weights.sum() * num_classes
    elif config.mode == "manual":
        if config.class_weights is None:
            raise ValueError("manual mode requires class_weights to be set")
        class_weights = torch.ones(num_classes)
        for cls_name, weight in config.class_weights.items():
            if cls_name not in class_to_idx:
                raise ValueError(f"class_weights key '{cls_name}' not in class_to_idx")
            class_weights[class_to_idx[cls_name]] = weight
    else:
        raise ValueError(f"Unknown sampler mode: {config.mode}")

    sample_weights = [class_weights[label].item() for label in labels]
    return TrackingWeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=config.replacement,
    )


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
