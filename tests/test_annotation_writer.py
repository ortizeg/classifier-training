"""Tests for ClassificationAnnotationWriter."""

from __future__ import annotations

import json
from pathlib import Path

from classifier_training.io.annotation import ClassificationAnnotationWriter
from classifier_training.schemas.annotation import (
    ClassificationAnnotation,
    ClassificationPrediction,
)
from classifier_training.schemas.info import AnnotationInfo


def _make_annotation(filename: str = "img_001.jpg") -> ClassificationAnnotation:
    return ClassificationAnnotation(
        filename=filename,
        categories={0: "", 1: "23"},
        info=AnnotationInfo(annotations_source="test"),
        ground_truth="23",
        predictions=[
            ClassificationPrediction(class_id=1, label="23", confidence=0.95),
        ],
    )


class TestClassificationAnnotationWriter:
    def test_write_creates_file(self, tmp_path: Path) -> None:
        writer = ClassificationAnnotationWriter(tmp_path / "out")
        ann = _make_annotation()
        out_path = writer.write(ann)
        assert out_path.exists()
        assert out_path.name == "img_001.json"

    def test_write_valid_json(self, tmp_path: Path) -> None:
        writer = ClassificationAnnotationWriter(tmp_path / "out")
        ann = _make_annotation()
        out_path = writer.write(ann)
        data = json.loads(out_path.read_text())
        assert data["filename"] == "img_001.jpg"
        assert data["ground_truth"] == "23"
        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["label"] == "23"

    def test_write_creates_output_dir(self, tmp_path: Path) -> None:
        deep_dir = tmp_path / "a" / "b" / "c"
        writer = ClassificationAnnotationWriter(deep_dir)
        ann = _make_annotation()
        out_path = writer.write(ann)
        assert out_path.exists()

    def test_write_multiple(self, tmp_path: Path) -> None:
        writer = ClassificationAnnotationWriter(tmp_path / "out")
        writer.write(_make_annotation("a.jpg"))
        writer.write(_make_annotation("b.png"))
        files = list((tmp_path / "out").glob("*.json"))
        assert len(files) == 2
        names = {f.stem for f in files}
        assert names == {"a", "b"}
