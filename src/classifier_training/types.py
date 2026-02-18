"""Type aliases and TypedDicts for classifier_training inter-module contracts."""

from typing import TypedDict

import torch


class ClassificationBatch(TypedDict):
    """A single batch from a classification DataLoader.

    images: Float tensor of shape (B, C, H, W), normalized with ImageNet stats.
    labels: Long tensor of shape (B,), integer class indices.
    """

    images: torch.Tensor
    labels: torch.Tensor
