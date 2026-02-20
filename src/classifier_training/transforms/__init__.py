"""Torchvision v2 transform pipeline for image classification training.

Custom ``v2.Transform`` subclasses that complement the standard torchvision v2
transforms.  All classes are designed to be fully parameterizable via Hydra YAML
configs and composable into train/val/test pipelines.
"""

from classifier_training.transforms.conversion import ToFloat32Tensor
from classifier_training.transforms.degradation import (
    RandomJPEGCompression,
    RandomPixelate,
)

__all__ = [
    "RandomJPEGCompression",
    "RandomPixelate",
    "ToFloat32Tensor",
]
