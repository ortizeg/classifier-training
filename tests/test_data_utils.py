"""Tests for data utility functions."""

import json
from pathlib import Path

import pytest
from PIL import Image

from classifier_training.data.utils import get_files


class TestGetFiles:
    def test_finds_files_with_matching_extension(self, tmp_path: Path) -> None:
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.txt").touch()

        result = get_files(tmp_path, (".jpg", ".png"))
        names = [p.name for p in result]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names

    def test_recurses_into_subdirectories(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.jsonl").touch()
        (sub / "deep.jsonl").touch()

        result = get_files(tmp_path, (".jsonl",))
        assert len(result) == 2
        names = [p.name for p in result]
        assert "top.jsonl" in names
        assert "deep.jsonl" in names

    def test_returns_sorted_paths(self, tmp_path: Path) -> None:
        (tmp_path / "c.txt").touch()
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()

        result = get_files(tmp_path, (".txt",))
        assert result == sorted(result)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        result = get_files(tmp_path, (".jsonl",))
        assert result == []

    def test_case_insensitive_extension_match(self, tmp_path: Path) -> None:
        (tmp_path / "img.JPG").touch()
        (tmp_path / "img.Png").touch()

        result = get_files(tmp_path, (".jpg", ".png"))
        assert len(result) == 2


class TestRecursiveDatasetDiscovery:
    """Test that JerseyNumberDataset discovers JSONL files recursively."""

    @pytest.fixture()
    def recursive_dataset_dir(self, tmp_path: Path) -> Path:
        """Dataset with annotations split across subdirectories."""
        classes = ["0", "1", "2"]

        for split in ("train", "valid", "test"):
            # Real data at top level
            split_dir = tmp_path / split
            split_dir.mkdir()
            lines: list[str] = []
            for cls in classes:
                fname = f"img_{cls}_00.jpg"
                img = Image.new("RGB", (224, 224), color=(int(cls) * 80, 100, 150))
                img.save(split_dir / fname)
                lines.append(
                    json.dumps(
                        {"image": fname, "prefix": "Read the number.", "suffix": cls}
                    )
                )
            (split_dir / "annotations.jsonl").write_text("\n".join(lines) + "\n")

            # Synthetic data in subdirectory (only for train)
            if split == "train":
                synth_dir = split_dir / "synthetic"
                synth_dir.mkdir()
                synth_lines: list[str] = []
                for cls in classes:
                    fname = f"synth_{cls}_00.jpg"
                    img = Image.new("RGB", (96, 96), color=(int(cls) * 60, 80, 120))
                    img.save(synth_dir / fname)
                    synth_lines.append(
                        json.dumps(
                            {
                                "image": fname,
                                "prefix": "Read the number.",
                                "suffix": cls,
                            }
                        )
                    )
                (synth_dir / "annotations.jsonl").write_text(
                    "\n".join(synth_lines) + "\n"
                )

        return tmp_path

    def test_dataset_discovers_multiple_jsonl_files(
        self, recursive_dataset_dir: Path
    ) -> None:
        from classifier_training.data.dataset import JerseyNumberDataset

        class_to_idx = {"0": 0, "1": 1, "2": 2}
        ds = JerseyNumberDataset(recursive_dataset_dir / "train", class_to_idx)
        # 3 real + 3 synthetic = 6 samples
        assert len(ds) == 6

    def test_dataset_resolves_images_relative_to_annotation(
        self, recursive_dataset_dir: Path
    ) -> None:
        from classifier_training.data.dataset import JerseyNumberDataset

        class_to_idx = {"0": 0, "1": 1, "2": 2}
        ds = JerseyNumberDataset(recursive_dataset_dir / "train", class_to_idx)
        # Check that synthetic paths point to the synthetic subdirectory
        synth_paths = [p for p, _ in ds.samples if "synthetic" in str(p)]
        assert len(synth_paths) == 3
        for p in synth_paths:
            assert p.exists()

    def test_backward_compat_single_jsonl(self, tmp_path: Path) -> None:
        """A directory with a single annotations.jsonl still works."""
        from classifier_training.data.dataset import JerseyNumberDataset

        split_dir = tmp_path / "single"
        split_dir.mkdir()
        lines = []
        for i in range(3):
            fname = f"img_{i}.jpg"
            Image.new("RGB", (64, 64)).save(split_dir / fname)
            lines.append(
                json.dumps(
                    {"image": fname, "prefix": "Read the number.", "suffix": str(i)}
                )
            )
        (split_dir / "annotations.jsonl").write_text("\n".join(lines) + "\n")

        class_to_idx = {"0": 0, "1": 1, "2": 2}
        ds = JerseyNumberDataset(split_dir, class_to_idx)
        assert len(ds) == 3

    def test_datamodule_class_to_idx_discovers_recursive(
        self, recursive_dataset_dir: Path
    ) -> None:
        """DataModule _build_class_to_idx finds classes from all JSONL files."""
        from classifier_training.config import DataModuleConfig
        from classifier_training.data import ImageFolderDataModule

        cfg = DataModuleConfig(data_root=str(recursive_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        c2i = dm.class_to_idx
        assert c2i == {"0": 0, "1": 1, "2": 2}

    def test_datamodule_setup_fit_with_recursive_data(
        self, recursive_dataset_dir: Path
    ) -> None:
        """setup('fit') loads train samples from all annotation files."""
        from classifier_training.config import DataModuleConfig
        from classifier_training.data import ImageFolderDataModule

        cfg = DataModuleConfig(data_root=str(recursive_dataset_dir), num_workers=0)
        dm = ImageFolderDataModule(cfg)
        dm.setup("fit")
        assert dm._train_dataset is not None
        assert len(dm._train_dataset) == 6  # 3 real + 3 synthetic
        assert dm._val_dataset is not None
        assert len(dm._val_dataset) == 3  # val has no synthetic subdir
