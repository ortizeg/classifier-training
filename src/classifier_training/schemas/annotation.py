"""Classification annotation schema.

Mirrors the DetectionAnnotation pattern from object-detection-training:
one annotation per image with top-K predictions sorted by confidence.
"""

from __future__ import annotations

from pydantic import BaseModel

from classifier_training.schemas.info import AnnotationInfo


class ClassificationPrediction(BaseModel, frozen=True):
    """A single classification prediction."""

    class_id: int
    label: str
    confidence: float


class ClassificationAnnotation(BaseModel):
    """Full annotation for a single image.

    Self-contained: includes the category mapping so each annotation
    file can be interpreted independently.
    """

    filename: str
    categories: dict[int, str]
    info: AnnotationInfo
    ground_truth: str | None = None
    predictions: list[ClassificationPrediction]
