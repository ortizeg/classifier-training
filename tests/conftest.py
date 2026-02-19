"""Shared pytest fixtures for classifier_training tests."""

import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def tmp_dataset_dir(tmp_path: Path) -> Path:
    """Minimal dataset in JSONL-annotated format for testing.

    Structure mirrors the real basketball-jersey-numbers-ocr dataset:
    - Flat image files per split (no class subdirectories)
    - annotations.jsonl with {"image": ..., "prefix": ..., "suffix": ...} per line

    3 splits x 3 classes x 2 images = 18 images total.
    Classes: "0", "1", "2" — 3 classes, all present in all splits.
    Train has 2 images per class (6 total) to allow sampler weight computation.
    """
    classes = ["0", "1", "2"]

    for split in ("train", "valid", "test"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        lines: list[str] = []

        for cls in classes:
            for i in range(2):
                fname = f"img_{cls}_{i:02d}.jpg"
                img = Image.new("RGB", (224, 224), color=(int(cls) * 80, 100, 150))
                img.save(split_dir / fname)
                lines.append(
                    json.dumps(
                        {
                            "image": fname,
                            "prefix": "Read the number.",
                            "suffix": cls,
                        }
                    )
                )

        (split_dir / "annotations.jsonl").write_text("\n".join(lines) + "\n")

    return tmp_path
