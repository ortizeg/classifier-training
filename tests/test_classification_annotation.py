"""Tests for ClassificationAnnotation and related schemas."""

from __future__ import annotations

from classifier_training.schemas.annotation import (
    ClassificationAnnotation,
    ClassificationPrediction,
)
from classifier_training.schemas.info import AnnotationInfo


def _make_annotation() -> ClassificationAnnotation:
    return ClassificationAnnotation(
        filename="img_001.jpg",
        categories={0: "", 1: "0", 2: "00", 3: "1"},
        info=AnnotationInfo(
            annotations_source="onnx-resnet18",
            image_width=100,
            image_height=150,
        ),
        ground_truth="1",
        predictions=[
            ClassificationPrediction(class_id=3, label="1", confidence=0.85),
            ClassificationPrediction(class_id=1, label="0", confidence=0.10),
            ClassificationPrediction(class_id=2, label="00", confidence=0.05),
        ],
    )


class TestClassificationPrediction:
    def test_frozen(self) -> None:
        pred = ClassificationPrediction(class_id=1, label="0", confidence=0.9)
        assert pred.class_id == 1
        assert pred.label == "0"
        assert pred.confidence == 0.9

    def test_immutable(self) -> None:
        import pytest

        pred = ClassificationPrediction(class_id=1, label="0", confidence=0.9)
        with pytest.raises(Exception):  # noqa: B017
            pred.class_id = 2  # type: ignore[misc]


class TestAnnotationInfo:
    def test_defaults(self) -> None:
        info = AnnotationInfo(annotations_source="test")
        assert info.annotations_source == "test"
        assert info.image_width is None
        assert info.image_height is None
        assert info.created_at  # non-empty string

    def test_with_dimensions(self) -> None:
        info = AnnotationInfo(
            annotations_source="test",
            image_width=224,
            image_height=224,
        )
        assert info.image_width == 224
        assert info.image_height == 224


class TestClassificationAnnotation:
    def test_roundtrip_json(self) -> None:
        ann = _make_annotation()
        data = ann.model_dump()
        restored = ClassificationAnnotation(**data)
        assert restored.filename == ann.filename
        assert restored.ground_truth == ann.ground_truth
        assert len(restored.predictions) == 3
        assert restored.predictions[0].confidence == 0.85

    def test_no_ground_truth(self) -> None:
        ann = ClassificationAnnotation(
            filename="unknown.jpg",
            categories={0: "a"},
            info=AnnotationInfo(annotations_source="test"),
            predictions=[],
        )
        assert ann.ground_truth is None

    def test_categories_int_keys(self) -> None:
        ann = _make_annotation()
        assert isinstance(next(iter(ann.categories.keys())), int)

    def test_predictions_sorted(self) -> None:
        ann = _make_annotation()
        confidences = [p.confidence for p in ann.predictions]
        assert confidences == sorted(confidences, reverse=True)
