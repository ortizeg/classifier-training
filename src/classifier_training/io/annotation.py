"""Classification annotation writer using orjson."""

from __future__ import annotations

from pathlib import Path

import orjson

from classifier_training.schemas.annotation import ClassificationAnnotation


class ClassificationAnnotationWriter:
    """Write one JSON file per image annotation using orjson.

    Output files are named ``{image_stem}.json`` inside ``output_dir``.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, annotation: ClassificationAnnotation) -> Path:
        """Write a single annotation to disk. Returns the output path."""
        stem = Path(annotation.filename).stem
        out_path = self.output_dir / f"{stem}.json"
        dump = annotation.model_dump()
        # orjson requires str dict keys; categories uses int keys
        dump["categories"] = {str(k): v for k, v in dump["categories"].items()}
        data = orjson.dumps(dump, option=orjson.OPT_INDENT_2)
        out_path.write_bytes(data)
        return out_path
