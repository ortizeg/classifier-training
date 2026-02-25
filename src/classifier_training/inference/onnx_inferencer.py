"""ONNX-based classification inferencer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from classifier_training.inference.base import BaseClassificationInferencer
from classifier_training.schemas.annotation import ClassificationPrediction


def _softmax(logits: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """Row-wise softmax for 2-D array."""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class ONNXClassificationInferencer(BaseClassificationInferencer):
    """Run classification inference with an ONNX model.

    Loads a model exported via :class:`ONNXExportCallback` together with
    its ``labels_mapping.json`` sidecar.  Uses the same val transforms
    as training (Resize 256 -> CenterCrop 224 -> ImageNet normalisation).

    Args:
        model_path: Path to the ``.onnx`` file.
        labels_mapping_path: Path to the ``labels_mapping.json`` sidecar.
        top_k: Number of top predictions to return.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_mapping_path: str | Path,
        top_k: int = 5,
    ) -> None:
        model_path = Path(model_path)
        labels_mapping_path = Path(labels_mapping_path)

        with open(labels_mapping_path) as f:
            mapping = json.load(f)

        self.idx_to_class: dict[int, str] = {
            int(k): v for k, v in mapping["idx_to_class"].items()
        }
        norm = mapping["normalization"]
        self.top_k = top_k

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm["mean"], std=norm["std"]),
            ]
        )

        self.session = ort.InferenceSession(
            str(model_path),
            providers=ort.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Image.Image) -> list[ClassificationPrediction]:
        """Single image inference."""
        tensor = self.transform(image.convert("RGB"))
        input_array = tensor.unsqueeze(0).numpy()  # type: ignore[union-attr]
        logits = self.session.run(None, {self.input_name: input_array})[0]
        return self._logits_to_predictions(logits[0])

    def predict_batch(
        self, images: list[Image.Image]
    ) -> list[list[ClassificationPrediction]]:
        """Batched inference — stack preprocessed images into one session.run()."""
        if not images:
            return []
        tensors = [self.transform(img.convert("RGB")) for img in images]
        import torch

        batch = torch.stack(tensors).numpy()  # type: ignore[arg-type]
        logits = self.session.run(None, {self.input_name: batch})[0]
        return [self._logits_to_predictions(logits[i]) for i in range(len(images))]

    def _logits_to_predictions(
        self,
        logits: np.ndarray,  # type: ignore[type-arg]
    ) -> list[ClassificationPrediction]:
        """Convert raw logits to sorted top-K predictions."""
        probs = _softmax(logits[np.newaxis, :])[0]
        top_indices = np.argsort(probs)[::-1][: self.top_k]
        return [
            ClassificationPrediction(
                class_id=int(idx),
                label=self.idx_to_class.get(int(idx), str(idx)),
                confidence=float(probs[idx]),
            )
            for idx in top_indices
        ]
