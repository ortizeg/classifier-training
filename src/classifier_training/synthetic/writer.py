"""Write synthetic jersey number images and JSONL annotations to disk."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from PIL import Image


class SyntheticWriter:
    """Writes synthetic images + annotations.jsonl to an output directory.

    Args:
        output_dir: Directory to write images and annotations.jsonl into.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, str]] = []

    def write_image(self, img: Image.Image, label: str, index: int) -> None:
        """Save a synthetic image and record its annotation.

        Args:
            img: PIL Image to save.
            label: Class label string (e.g. "6", "46").
            index: Sample index within this label (for filename uniqueness).
        """
        # Zero-pad label for filename sorting (e.g. "6" -> "006")
        label_padded = label.zfill(3) if label else "empty"
        fname = f"synth_{label_padded}_{index:05d}.jpg"
        img.save(self.output_dir / fname, quality=95)
        self._records.append(
            {
                "image": fname,
                "prefix": "Read the number.",
                "suffix": label,
            }
        )

    def flush(self) -> Path:
        """Write annotations.jsonl and return its path.

        Returns:
            Path to the written annotations.jsonl file.
        """
        ann_path = self.output_dir / "annotations.jsonl"
        with open(ann_path, "w") as f:
            for record in self._records:
                f.write(json.dumps(record) + "\n")
        logger.info(f"Wrote {len(self._records)} synthetic annotations to {ann_path}")
        return ann_path

    @property
    def num_written(self) -> int:
        """Number of images written so far."""
        return len(self._records)
