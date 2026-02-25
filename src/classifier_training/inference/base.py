"""Abstract base class for classification inferencers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from classifier_training.schemas.annotation import ClassificationPrediction


class BaseClassificationInferencer(ABC):
    """Base class for classification inferencers.

    Subclasses must implement ``predict`` (single image) and
    ``predict_batch`` (multiple images).  Both return predictions
    sorted by descending confidence.
    """

    @abstractmethod
    def predict(self, image: Image.Image) -> list[ClassificationPrediction]:
        """Run inference on a single image.

        Returns predictions sorted by confidence descending.
        """

    @abstractmethod
    def predict_batch(
        self, images: list[Image.Image]
    ) -> list[list[ClassificationPrediction]]:
        """Run inference on a batch of images.

        Returns a list of prediction lists, one per image,
        each sorted by confidence descending.
        """
