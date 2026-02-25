"""Classification inference framework."""

from classifier_training.inference.base import BaseClassificationInferencer
from classifier_training.inference.onnx_inferencer import (
    ONNXClassificationInferencer,
)

__all__ = [
    "BaseClassificationInferencer",
    "ONNXClassificationInferencer",
]
