"""Broadcast video degradation transforms for classification training.

Simulates JPEG compression artifacts and low-resolution pixelation commonly
seen in broadcast basketball footage.  Both transforms operate on PIL images
and should be placed *before* ``ToFloat32Tensor`` in the pipeline.
"""

from __future__ import annotations

import io
import random
from typing import Any

from PIL import Image
from torchvision.transforms import v2


class RandomJPEGCompression(v2.Transform):
    """Simulate JPEG encoding artifacts via a PIL encode/decode round-trip.

    Args:
        quality_min: Minimum JPEG quality (lower = more artifacts).
        quality_max: Maximum JPEG quality.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        quality_min: int = 20,
        quality_max: int = 75,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        if not 1 <= quality_min <= quality_max <= 100:
            raise ValueError(
                f"quality_min ({quality_min}) and quality_max ({quality_max}) "
                "must satisfy 1 <= quality_min <= quality_max <= 100"
            )
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        img = inputs[0]
        rest = inputs[1:]

        if not isinstance(img, Image.Image):
            raise TypeError(
                f"RandomJPEGCompression expects a PIL Image, got {type(img)}"
            )

        if random.random() >= self.p:  # noqa: S311
            return inputs if rest else img

        quality = random.randint(self.quality_min, self.quality_max)  # noqa: S311
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        compressed = Image.open(buf).convert(img.mode)

        return (compressed, *rest) if rest else compressed


class RandomPixelate(v2.Transform):
    """Simulate low-resolution broadcast crops via downscale + upscale.

    Downscales to ``original_size * scale`` then upscales back to the original
    size, both with ``NEAREST`` interpolation to create blocky pixelation.

    Args:
        scale_min: Minimum downscale factor (e.g. 0.25 = quarter resolution).
        scale_max: Maximum downscale factor.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        scale_min: float = 0.25,
        scale_max: float = 0.75,
        p: float = 0.3,
    ) -> None:
        super().__init__()
        if not 0.0 < scale_min <= scale_max < 1.0:
            raise ValueError(
                f"scale_min ({scale_min}) and scale_max ({scale_max}) "
                "must satisfy 0 < scale_min <= scale_max < 1"
            )
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        img = inputs[0]
        rest = inputs[1:]

        if not isinstance(img, Image.Image):
            raise TypeError(f"RandomPixelate expects a PIL Image, got {type(img)}")

        if random.random() >= self.p:  # noqa: S311
            return inputs if rest else img

        scale = random.uniform(self.scale_min, self.scale_max)  # noqa: S311
        orig_size = img.size  # (width, height)
        small_size = (
            max(1, int(orig_size[0] * scale)),
            max(1, int(orig_size[1] * scale)),
        )

        pixelated = img.resize(small_size, Image.NEAREST).resize(
            orig_size, Image.NEAREST
        )

        return (pixelated, *rest) if rest else pixelated
