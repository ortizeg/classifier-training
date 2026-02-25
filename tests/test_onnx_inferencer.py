"""Tests for ONNXClassificationInferencer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from classifier_training.inference.onnx_inferencer import (
    ONNXClassificationInferencer,
    _softmax,
)


def _make_labels_mapping(tmp_path: Path) -> Path:
    """Create a minimal labels_mapping.json."""
    mapping = {
        "num_classes": 3,
        "class_to_idx": {"": 0, "1": 1, "23": 2},
        "idx_to_class": {"0": "", "1": "1", "2": "23"},
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    path = tmp_path / "labels_mapping.json"
    path.write_text(json.dumps(mapping))
    return path


def _make_dummy_image() -> Image.Image:
    return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0]])
        probs = _softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_correct_ordering(self) -> None:
        logits = np.array([[1.0, 3.0, 2.0]])
        probs = _softmax(logits)
        assert probs[0, 1] > probs[0, 2] > probs[0, 0]


class TestONNXClassificationInferencer:
    @patch("classifier_training.inference.onnx_inferencer.ort.InferenceSession")
    def test_predict_returns_sorted(
        self, mock_session_cls: MagicMock, tmp_path: Path
    ) -> None:
        labels_path = _make_labels_mapping(tmp_path)

        # Mock ONNX session
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        # Return logits: class 2 ("23") is highest
        mock_session.run.return_value = [np.array([[0.1, 0.5, 2.0]])]
        mock_session_cls.return_value = mock_session

        model_path = tmp_path / "model.onnx"
        model_path.touch()

        inferencer = ONNXClassificationInferencer(
            model_path=model_path,
            labels_mapping_path=labels_path,
            top_k=3,
        )

        image = _make_dummy_image()
        preds = inferencer.predict(image)

        assert len(preds) == 3
        assert preds[0].label == "23"
        assert preds[0].confidence > preds[1].confidence

    @patch("classifier_training.inference.onnx_inferencer.ort.InferenceSession")
    def test_predict_batch(self, mock_session_cls: MagicMock, tmp_path: Path) -> None:
        labels_path = _make_labels_mapping(tmp_path)

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        # Batch of 2 images
        mock_session.run.return_value = [np.array([[2.0, 0.1, 0.1], [0.1, 0.1, 2.0]])]
        mock_session_cls.return_value = mock_session

        model_path = tmp_path / "model.onnx"
        model_path.touch()

        inferencer = ONNXClassificationInferencer(
            model_path=model_path,
            labels_mapping_path=labels_path,
            top_k=2,
        )

        images = [_make_dummy_image(), _make_dummy_image()]
        batch_preds = inferencer.predict_batch(images)

        assert len(batch_preds) == 2
        # First image: class 0 ("") is highest
        assert batch_preds[0][0].label == ""
        # Second image: class 2 ("23") is highest
        assert batch_preds[1][0].label == "23"

    @patch("classifier_training.inference.onnx_inferencer.ort.InferenceSession")
    def test_predict_batch_empty(
        self, mock_session_cls: MagicMock, tmp_path: Path
    ) -> None:
        labels_path = _make_labels_mapping(tmp_path)

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session_cls.return_value = mock_session

        model_path = tmp_path / "model.onnx"
        model_path.touch()

        inferencer = ONNXClassificationInferencer(
            model_path=model_path,
            labels_mapping_path=labels_path,
        )

        assert inferencer.predict_batch([]) == []
