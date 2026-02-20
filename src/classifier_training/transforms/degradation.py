"""Broadcast video degradation transforms for classification training.

Simulates JPEG compression artifacts, low-resolution pixelation, sensor noise,
and bilinear resize blur commonly seen in broadcast basketball footage.  All
transforms operate on PIL images and should be placed *before*
``ToFloat32Tensor`` in the pipeline.
"""

from __future__ import annotations

import io
import random
from typing import Any

import numpy as np
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


class RandomGaussianNoise(v2.Transform):
    """Add random Gaussian noise to simulate sensor noise in dim arenas.

    Args:
        sigma_min: Minimum noise standard deviation (pixel scale 0-255).
        sigma_max: Maximum noise standard deviation.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        sigma_min: float = 5.0,
        sigma_max: float = 25.0,
        p: float = 0.3,
    ) -> None:
        super().__init__()
        if not 0.0 < sigma_min <= sigma_max:
            raise ValueError(
                f"sigma_min ({sigma_min}) and sigma_max ({sigma_max}) "
                "must satisfy 0 < sigma_min <= sigma_max"
            )
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        img = inputs[0]
        rest = inputs[1:]

        if not isinstance(img, Image.Image):
            raise TypeError(f"RandomGaussianNoise expects a PIL Image, got {type(img)}")

        if random.random() >= self.p:  # noqa: S311
            return inputs if rest else img

        sigma = random.uniform(self.sigma_min, self.sigma_max)  # noqa: S311
        arr = np.array(img, dtype=np.float32)
        noise = (
            np.random.default_rng()
            .normal(0.0, sigma, arr.shape)
            .astype(np.float32)
        )
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        result = Image.fromarray(noisy, mode=img.mode)

        return (result, *rest) if rest else result


class RandomBilinearDownscale(v2.Transform):
    """Simulate smooth broadcast resize via bilinear downscale + upscale.

    Unlike ``RandomPixelate`` (which uses NEAREST for blocky artifacts), this
    uses ``BILINEAR`` interpolation producing softer, more realistic broadcast
    resize blur.

    Args:
        scale_min: Minimum downscale factor (e.g. 0.3 = 30% resolution).
        scale_max: Maximum downscale factor.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        scale_min: float = 0.3,
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
            raise TypeError(
                f"RandomBilinearDownscale expects a PIL Image, got {type(img)}"
            )

        if random.random() >= self.p:  # noqa: S311
            return inputs if rest else img

        scale = random.uniform(self.scale_min, self.scale_max)  # noqa: S311
        orig_size = img.size  # (width, height)
        small_size = (
            max(1, int(orig_size[0] * scale)),
            max(1, int(orig_size[1] * scale)),
        )

        blurred = img.resize(small_size, Image.BILINEAR).resize(
            orig_size, Image.BILINEAR
        )

        return (blurred, *rest) if rest else blurred


class RandomZoomOut(v2.Transform):
    """Simulate a loose detector bounding box by zooming out with padding.

    Places the image onto a larger canvas filled with ``fill_color``, then
    resizes back to the original dimensions.  This makes the subject appear
    smaller within the crop, as happens when an object detector returns a box
    that is larger than the actual object.

    Args:
        max_scale: Maximum zoom-out factor.  A value of 1.5 means up to 50%
            extra context on each side.  Must be > 1.0.
        min_scale: Minimum zoom-out factor.  1.0 = no zoom-out.
        fill_color: RGB tuple used to fill the padding area.  Defaults to
            ImageNet mean (124, 116, 104).
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        min_scale: float = 1.0,
        max_scale: float = 1.5,
        fill_color: tuple[int, int, int] = (124, 116, 104),
        p: float = 0.3,
    ) -> None:
        super().__init__()
        if not 1.0 <= min_scale <= max_scale:
            raise ValueError(
                f"min_scale ({min_scale}) and max_scale ({max_scale}) "
                "must satisfy 1.0 <= min_scale <= max_scale"
            )
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.fill_color = fill_color
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        img = inputs[0]
        rest = inputs[1:]

        if not isinstance(img, Image.Image):
            raise TypeError(
                f"RandomZoomOut expects a PIL Image, got {type(img)}"
            )

        if random.random() >= self.p:  # noqa: S311
            return inputs if rest else img

        scale = random.uniform(self.min_scale, self.max_scale)  # noqa: S311
        if scale <= 1.0:
            return inputs if rest else img

        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)

        canvas = Image.new(img.mode, (new_w, new_h), self.fill_color)

        # Random placement of original image on the larger canvas
        max_x = new_w - w
        max_y = new_h - h
        offset_x = random.randint(0, max_x) if max_x > 0 else 0  # noqa: S311
        offset_y = random.randint(0, max_y) if max_y > 0 else 0  # noqa: S311
        canvas.paste(img, (offset_x, offset_y))

        # Resize back to original dimensions
        result = canvas.resize((w, h), Image.BILINEAR)

        return (result, *rest) if rest else result


class RandomZoomIn(v2.Transform):
    """Simulate a tight detector bounding box by cropping and resizing up.

    Crops a random sub-region of the image and resizes it back to the
    original dimensions, making the subject appear larger and potentially
    cutting off edges — as happens when an object detector returns a box
    that is smaller than the full object.

    Args:
        min_scale: Minimum crop fraction (e.g. 0.7 = crop to 70% of image).
            Must be in (0, 1].
        max_scale: Maximum crop fraction.  1.0 = no crop.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        min_scale: float = 0.7,
        max_scale: float = 1.0,
        p: float = 0.3,
    ) -> None:
        super().__init__()
        if not 0.0 < min_scale <= max_scale <= 1.0:
            raise ValueError(
                f"min_scale ({min_scale}) and max_scale ({max_scale}) "
                "must satisfy 0 < min_scale <= max_scale <= 1.0"
            )
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        img = inputs[0]
        rest = inputs[1:]

        if not isinstance(img, Image.Image):
            raise TypeError(
                f"RandomZoomIn expects a PIL Image, got {type(img)}"
            )

        if random.random() >= self.p:  # noqa: S311
            return inputs if rest else img

        scale = random.uniform(self.min_scale, self.max_scale)  # noqa: S311
        if scale >= 1.0:
            return inputs if rest else img

        w, h = img.size
        crop_w, crop_h = int(w * scale), int(h * scale)

        # Random offset for the crop
        max_x = w - crop_w
        max_y = h - crop_h
        x = random.randint(0, max_x) if max_x > 0 else 0  # noqa: S311
        y = random.randint(0, max_y) if max_y > 0 else 0  # noqa: S311

        cropped = img.crop((x, y, x + crop_w, y + crop_h))
        result = cropped.resize((w, h), Image.BILINEAR)

        return (result, *rest) if rest else result
