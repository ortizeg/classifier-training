"""Classification model implementations."""

import classifier_training.models.smolvlm2  # noqa: F401  # register with Hydra
from classifier_training.models.base import BaseClassificationModel
from classifier_training.models.resnet import (
    ResNet18ClassificationModel,
    ResNet34ClassificationModel,
    ResNet50ClassificationModel,
)

__all__ = [
    "BaseClassificationModel",
    "ResNet18ClassificationModel",
    "ResNet34ClassificationModel",
    "ResNet50ClassificationModel",
]
