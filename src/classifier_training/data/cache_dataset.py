"""Cached dataset wrapper for faster training data loading.

Wraps a JerseyNumberDataset and caches decoded PIL images + labels.
Supports two cache backends:

- **ram**: Stores samples in a Python list for zero-overhead reads.
- **disk**: Persists samples in a SQLite database for cross-run reuse.

Cache stores **pre-transform** data so stochastic augmentations still
produce different results each epoch.
"""

from __future__ import annotations

import concurrent.futures
import copy
import io
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import psutil  # type: ignore[import-untyped]
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm

from classifier_training.data.dataset import JerseyNumberDataset

__all__ = ["CacheDataset"]

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cache (
    idx       INTEGER PRIMARY KEY,
    img_blob  BLOB    NOT NULL,
    label     INTEGER NOT NULL
);
"""

_INSERT = """
INSERT OR REPLACE INTO cache (idx, img_blob, label)
VALUES (?, ?, ?);
"""

_SELECT = "SELECT img_blob, label FROM cache WHERE idx = ?;"

_COUNT = "SELECT COUNT(*) FROM cache;"

# JPEG quality for compressed storage (95 is visually lossless)
_JPEG_QUALITY = 95


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _serialize_image(img: Image.Image) -> bytes:
    """Serialize a PIL Image to compressed JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()


def _deserialize_image(img_bytes: bytes) -> Image.Image:
    """Reconstruct a PIL Image from JPEG bytes."""
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ---------------------------------------------------------------------------
# CacheDataset
# ---------------------------------------------------------------------------
class CacheDataset(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    """Wraps a JerseyNumberDataset with an in-memory or on-disk cache.

    On first access, every sample is read from the underlying dataset and
    cached. Subsequent reads bypass PIL I/O entirely.

    Parameters
    ----------
    dataset:
        The underlying JerseyNumberDataset.
    cache_type:
        - "ram": Store decoded images in memory (fastest, high RAM usage).
        - "disk": Store compressed images in SQLite (slower, low RAM usage).
        - "auto": Automatically select based on available system RAM.
    cache_dir:
        Directory for the SQLite DB (disk mode only). Defaults to
        ``{dataset.root}/.cache/``.
    transforms:
        Optional transforms applied **after** cache retrieval so that
        stochastic augmentations produce different results each epoch.
    rebuild:
        If True, delete any existing cache and rebuild from scratch.
    num_threads:
        Number of worker threads for parallel cache building.
    """

    def __init__(
        self,
        dataset: JerseyNumberDataset,
        cache_type: Literal["ram", "disk", "auto"] = "ram",
        cache_dir: str | Path | None = None,
        transforms: Callable[[Image.Image], torch.Tensor] | None = None,
        rebuild: bool = False,
        num_threads: int | None = None,
    ) -> None:
        self._dataset = dataset
        self.transforms = transforms

        if cache_type == "auto":
            self._cache_type = self._resolve_cache_type()
        else:
            self._cache_type = cache_type

        if num_threads is None:
            cpu_count = os.cpu_count() or 1
            self.num_threads = min(32, cpu_count + 4)
        else:
            self.num_threads = num_threads

        # --- RAM cache state ---
        self._ram_cache: list[tuple[Image.Image, int]] | None = None

        # --- Disk cache state ---
        self._db_path: Path | None = None
        self._conn: sqlite3.Connection | None = None

        if self._cache_type == "ram":
            self._build_ram_cache()
        else:
            if cache_dir is None:
                cache_dir = dataset.root / ".cache"
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = self._cache_dir / f"{dataset.root.name}.db"

            if rebuild and self._db_path.exists():
                logger.info(f"Removing existing cache: {self._db_path}")
                self._db_path.unlink()

            self._ensure_disk_cache()

    # ------------------------------------------------------------------
    # RAM cache
    # ------------------------------------------------------------------
    def _build_ram_cache(self) -> None:
        """Load all samples into a Python list for zero-overhead reads."""
        saved_transform = self._dataset.transform
        self._dataset.transform = None

        total = len(self._dataset)
        logger.info(f"Building RAM cache: {total} samples (threads={self.num_threads})")

        def _load_sample(idx: int) -> tuple[Image.Image, int]:
            img, label = self._dataset[idx]
            # img is PIL Image when transform is None
            return img, label  # type: ignore[return-value]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_threads
        ) as executor:
            results = list(
                tqdm(
                    executor.map(_load_sample, range(total)),
                    total=total,
                    desc="RAM Cache",
                    unit="img",
                )
            )

        self._ram_cache = results
        self._dataset.transform = saved_transform
        logger.info(f"RAM cache built: {total} samples in memory")

    # ------------------------------------------------------------------
    # SQLite connection management
    # ------------------------------------------------------------------
    @property
    def _connection(self) -> sqlite3.Connection:
        """Return a per-process SQLite connection (WAL mode for readers)."""
        if self._conn is None:
            if self._db_path is None:
                raise RuntimeError("No db_path configured for disk cache")
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        return self._conn

    # ------------------------------------------------------------------
    # Disk cache building
    # ------------------------------------------------------------------
    def _ensure_disk_cache(self) -> None:
        """Build the disk cache if it is missing or incomplete."""
        conn = self._connection
        conn.execute(_CREATE_TABLE)
        conn.commit()

        count = conn.execute(_COUNT).fetchone()[0]
        expected = len(self._dataset)

        if count >= expected:
            logger.info(f"Cache hit: {self._db_path} ({count} samples already cached)")
            return
        logger.info(
            f"Building cache: {self._db_path} "
            f"({count}/{expected} samples present, caching remaining)"
        )
        self._build_disk_cache()

    def _build_disk_cache(self) -> None:
        """Populate the SQLite cache from the underlying dataset."""
        saved_transform = self._dataset.transform
        self._dataset.transform = None

        conn = self._connection
        conn.execute(_CREATE_TABLE)

        total = len(self._dataset)
        batch_size = 1000

        existing_cursor = conn.execute("SELECT idx FROM cache")
        existing_indices = {row[0] for row in existing_cursor.fetchall()}
        indices_to_process = [i for i in range(total) if i not in existing_indices]

        if not indices_to_process:
            self._dataset.transform = saved_transform
            logger.info("All samples already cached.")
            return

        logger.info(
            f"Building disk cache: {len(indices_to_process)} samples "
            f"(threads={self.num_threads})"
        )

        def _process_sample(idx: int) -> tuple[int, bytes, int] | None:
            try:
                img, label = self._dataset[idx]
                img_bytes = _serialize_image(img)  # type: ignore[arg-type]
                return idx, img_bytes, label
            except Exception as e:
                logger.error(f"Failed to process sample {idx}: {e}")
                return None

        pending_inserts: list[tuple[int, bytes, int]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_threads
        ) as executor:
            futures = [
                executor.submit(_process_sample, idx) for idx in indices_to_process
            ]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(indices_to_process),
                desc="Disk Cache",
                unit="img",
            ):
                result = future.result()
                if result is None:
                    continue
                pending_inserts.append(result)
                if len(pending_inserts) >= batch_size:
                    conn.executemany(_INSERT, pending_inserts)
                    conn.commit()
                    pending_inserts.clear()

        if pending_inserts:
            conn.executemany(_INSERT, pending_inserts)
            conn.commit()

        self._dataset.transform = saved_transform
        logger.info(f"Cache built: {total} samples in {self._db_path}")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Retrieve a sample from cache and apply transforms."""
        if self._cache_type == "ram":
            return self._getitem_ram(idx)
        return self._getitem_disk(idx)

    def _getitem_ram(self, idx: int) -> tuple[torch.Tensor, int]:
        """Read from in-memory list (deepcopy to prevent mutation)."""
        if self._ram_cache is None:
            raise RuntimeError("RAM cache not initialized")
        img, label = copy.deepcopy(self._ram_cache[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label  # type: ignore[return-value]

    def _getitem_disk(self, idx: int) -> tuple[torch.Tensor, int]:
        """Read from SQLite disk cache."""
        row = self._connection.execute(_SELECT, (idx,)).fetchone()
        if row is None:
            raise RuntimeError(
                f"Cache miss at index {idx} — cache may be corrupted. "
                f"Delete {self._db_path} and re-run to rebuild."
            )
        img_bytes, label = row
        img = _deserialize_image(img_bytes)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Proxy properties for compatibility
    # ------------------------------------------------------------------
    @property
    def dataset(self) -> JerseyNumberDataset:
        """Access the underlying wrapped dataset."""
        return self._dataset

    @property
    def samples(self) -> list[tuple[Path, int]]:
        """Proxy to underlying dataset samples for sampler weight computation."""
        return self._dataset.samples

    @property
    def class_to_idx(self) -> dict[str, int]:
        return self._dataset.class_to_idx

    def __repr__(self) -> str:
        if self._cache_type == "ram":
            return (
                f"CacheDataset(wrapped={self._dataset!r}, "
                f"cache_type='ram', cached={len(self)} samples)"
            )
        return (
            f"CacheDataset(wrapped={self._dataset!r}, "
            f"cache_type='disk', cache={self._db_path}, "
            f"cached={len(self)} samples)"
        )

    def _close(self) -> None:
        """Close SQLite connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _resolve_cache_type(self) -> Literal["ram", "disk"]:
        """Determine whether to use RAM or disk cache based on available memory.

        Estimates uncompressed RAM usage from dataset size and image dimensions.
        Selects 'ram' if estimated usage is < 50% of available system RAM.
        """
        try:
            # Estimate: assume 224x224x3 per image (typical for classification)
            total = len(self._dataset)
            estimated_bytes = int(total * 224 * 224 * 3 * 1.2)  # 1.2x overhead

            available_bytes = psutil.virtual_memory().available
            threshold = available_bytes * 0.5

            logger.info(
                f"Auto-cache: Est. dataset size={estimated_bytes / 1e9:.2f}GB, "
                f"Available RAM={available_bytes / 1e9:.2f}GB, "
                f"Threshold={threshold / 1e9:.2f}GB"
            )

            if estimated_bytes < threshold:
                logger.info("Auto-cache: Selected 'ram' mode")
                return "ram"
            else:
                logger.info("Auto-cache: Selected 'disk' mode")
                return "disk"

        except Exception as e:
            logger.warning(f"Auto-cache dispatch failed: {e}. Defaulting to 'disk'.")
            return "disk"
