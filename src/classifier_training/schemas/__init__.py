"""Classification annotation schemas."""

from classifier_training.schemas.annotation import (
    ClassificationAnnotation,
    ClassificationPrediction,
)
from classifier_training.schemas.info import AnnotationInfo

__all__ = [
    "AnnotationInfo",
    "ClassificationAnnotation",
    "ClassificationPrediction",
]
