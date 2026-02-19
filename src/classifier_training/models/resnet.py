"""ResNet18, ResNet34, and ResNet50 classification models."""

from __future__ import annotations

from typing import Any

import torch
import torchvision.models as tv_models

from classifier_training.models.base import BaseClassificationModel
from classifier_training.utils.hydra import register


@register(name="resnet18", group="model")
class ResNet18ClassificationModel(BaseClassificationModel):
    """ResNet18 backbone with ImageNet pretrained weights.

    fc replaced with Linear(512, num_classes).
    Pass pretrained=False in tests to skip the ~44MB weight download.
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)  # type: ignore[no-any-return]


@register(name="resnet34", group="model")
class ResNet34ClassificationModel(BaseClassificationModel):
    """ResNet34 backbone with ImageNet pretrained weights.

    fc replaced with Linear(512, num_classes).
    Pass pretrained=False in tests to skip the ~84MB weight download.
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet34(weights=weights)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)  # type: ignore[no-any-return]


@register(name="resnet50", group="model")
class ResNet50ClassificationModel(BaseClassificationModel):
    """ResNet50 backbone with ImageNet pretrained weights.

    fc replaced with Linear(2048, num_classes).
    Pass pretrained=False in tests to skip the ~98MB weight download.
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_classes=num_classes, **kwargs)
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet50(weights=weights)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)  # type: ignore[no-any-return]
