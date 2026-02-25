"""Tests for synthetic renderer and writer with color overrides and metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from classifier_training.synthetic.renderer import (
    _COLOR_BY_NAME,
    JerseyNumberRenderer,
)
from classifier_training.synthetic.writer import SyntheticWriter

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"


@pytest.fixture()
def renderer() -> JerseyNumberRenderer:
    return JerseyNumberRenderer(font_dir=FONT_DIR, image_size=96, seed=0)


class TestRendererColorOverride:
    def test_returns_tuple(self, renderer: JerseyNumberRenderer) -> None:
        result = renderer.render("5")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_image_type(self, renderer: JerseyNumberRenderer) -> None:
        img, _ = renderer.render("5")
        assert isinstance(img, Image.Image)
        assert img.size == (96, 96)

    def test_metadata_keys(self, renderer: JerseyNumberRenderer) -> None:
        _, metadata = renderer.render("5")
        assert "jersey_color" in metadata
        assert "number_color" in metadata
        assert "border" in metadata

    def test_bg_override_in_metadata(self, renderer: JerseyNumberRenderer) -> None:
        _, metadata = renderer.render(
            "18", bg_color=(255, 255, 255), text_color=(206, 17, 38)
        )
        assert metadata["jersey_color"] == "white"
        assert metadata["number_color"] == "red"

    def test_text_override_orange(self, renderer: JerseyNumberRenderer) -> None:
        _, metadata = renderer.render(
            "4", bg_color=(255, 255, 255), text_color=(255, 103, 31)
        )
        assert metadata["number_color"] == "orange"

    def test_blue_bg_red_text(self, renderer: JerseyNumberRenderer) -> None:
        _, metadata = renderer.render(
            "22", bg_color=(0, 83, 188), text_color=(206, 17, 38)
        )
        assert metadata["jersey_color"] == "blue"
        assert metadata["number_color"] == "red"

    def test_border_always_true(self, renderer: JerseyNumberRenderer) -> None:
        _, metadata = renderer.render("7")
        assert metadata["border"] is True

    def test_random_path_unchanged(self, renderer: JerseyNumberRenderer) -> None:
        """No override -> render still works with random colors."""
        img, metadata = renderer.render("22")
        assert isinstance(img, Image.Image)
        assert metadata["jersey_color"] in _COLOR_BY_NAME

    def test_red_in_light_bg_pool(self) -> None:
        """With many renders on white bg, red should appear."""
        renderer = JerseyNumberRenderer(font_dir=FONT_DIR, seed=99)
        colors_seen: set[str] = set()
        for _ in range(50):
            _, meta = renderer.render("5", bg_color=(255, 255, 255))
            colors_seen.add(str(meta["number_color"]))
        assert "red" in colors_seen

    def test_orange_in_dark_bg_pool(self) -> None:
        """With many renders on black bg, orange should appear."""
        renderer = JerseyNumberRenderer(font_dir=FONT_DIR, seed=99)
        colors_seen: set[str] = set()
        for _ in range(50):
            _, meta = renderer.render("5", bg_color=(0, 0, 0))
            colors_seen.add(str(meta["number_color"]))
        assert "orange" in colors_seen


class TestWriterMetadata:
    def test_metadata_written_to_jsonl(self, tmp_path: Path) -> None:
        writer = SyntheticWriter(tmp_path / "out")
        img = Image.new("RGB", (96, 96), (255, 255, 255))
        metadata = {"jersey_color": "white", "number_color": "red", "border": True}
        writer.write_image(img, "18", 0, metadata=metadata)
        ann_path = writer.flush()
        records = [json.loads(line) for line in ann_path.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["metadata"]["jersey_color"] == "white"
        assert records[0]["metadata"]["number_color"] == "red"
        assert records[0]["metadata"]["border"] is True

    def test_no_metadata_omitted(self, tmp_path: Path) -> None:
        """When metadata=None, key should be absent."""
        writer = SyntheticWriter(tmp_path / "out")
        img = Image.new("RGB", (96, 96))
        writer.write_image(img, "5", 0)
        ann_path = writer.flush()
        record = json.loads(ann_path.read_text().strip())
        assert "metadata" not in record

    def test_backward_compat_no_metadata(self, tmp_path: Path) -> None:
        """Writer without metadata produces same format as before."""
        writer = SyntheticWriter(tmp_path / "out")
        img = Image.new("RGB", (96, 96))
        writer.write_image(img, "22", 0)
        ann_path = writer.flush()
        record = json.loads(ann_path.read_text().strip())
        assert record["image"] == "synth_022_00000.jpg"
        assert record["prefix"] == "Read the number."
        assert record["suffix"] == "22"
