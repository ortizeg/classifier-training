"""Torchvision v2 transform pipeline for image classification training.

Custom ``v2.Transform`` subclasses that complement the standard torchvision v2
transforms.  All classes are designed to be fully parameterizable via Hydra YAML
configs and composable into train/val/test pipelines.
"""

from classifier_training.transforms.conversion import ToFloat32Tensor
from classifier_training.transforms.degradation import (
    RandomBilinearDownscale,
    RandomGaussianNoise,
    RandomJPEGCompression,
    RandomPixelate,
    RandomZoomIn,
    RandomZoomOut,
)

__all__ = [
    "RandomBilinearDownscale",
    "RandomGaussianNoise",
    "RandomJPEGCompression",
    "RandomPixelate",
    "RandomZoomIn",
    "RandomZoomOut",
    "ToFloat32Tensor",
]
