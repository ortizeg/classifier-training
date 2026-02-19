"""Loss functions for classification training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Focal loss down-weights well-classified examples, focusing training
    on hard negatives.  When ``gamma=0`` this reduces to weighted
    cross-entropy with optional label smoothing.

    Parameters
    ----------
    gamma:
        Focusing parameter.  Higher values increase focus on hard examples.
    weight:
        Per-class weights (same semantics as ``CrossEntropyLoss.weight``).
    label_smoothing:
        Label smoothing factor in ``[0, 1)``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits:
            Raw model output of shape ``(B, C)``.
        targets:
            Ground-truth class indices of shape ``(B,)``.
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def build_loss_fn(
    name: str,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory for loss functions.

    Parameters
    ----------
    name:
        Loss function name: ``"cross_entropy"`` or ``"focal"``.
    weight:
        Per-class weights tensor.
    label_smoothing:
        Label smoothing factor.
    focal_gamma:
        Gamma for focal loss (ignored for cross_entropy).

    Returns
    -------
    nn.Module
        The configured loss function.
    """
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            weight=weight,
            label_smoothing=label_smoothing,
        )
    msg = f"Unknown loss function: {name!r}. Use 'cross_entropy' or 'focal'."
    raise ValueError(msg)
