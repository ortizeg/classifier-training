"""Synthetic jersey number renderer with broadcast degradation.

Renders digit strings onto colored backgrounds using sports fonts, applies
geometric distortion and broadcast-quality degradation transforms to produce
training images that match the visual domain of real basketball broadcast crops.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from classifier_training.transforms.degradation import (
    RandomBilinearDownscale,
    RandomGaussianNoise,
    RandomJPEGCompression,
)

# NBA jersey palette — common background colors
_JERSEY_BACKGROUNDS: list[tuple[int, int, int]] = [
    (255, 255, 255),  # white
    (0, 0, 0),  # black
    (0, 32, 78),  # navy
    (206, 17, 38),  # red
    (0, 83, 188),  # royal blue
    (253, 185, 39),  # gold
    (0, 122, 51),  # green
    (93, 14, 65),  # purple
    (128, 128, 128),  # grey
    (255, 103, 31),  # orange
    (196, 30, 58),  # maroon
]

# Named color lookup for targeted generation
_COLOR_BY_NAME: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "navy": (0, 32, 78),
    "red": (206, 17, 38),
    "blue": (0, 83, 188),
    "gold": (253, 185, 39),
    "green": (0, 122, 51),
    "purple": (93, 14, 65),
    "grey": (128, 128, 128),
    "orange": (255, 103, 31),
    "maroon": (196, 30, 58),
}

# Reverse lookup: RGB tuple → color name
_RGB_TO_NAME: dict[tuple[int, int, int], str] = {
    v: k for k, v in _COLOR_BY_NAME.items()
}


def _rgb_to_name(rgb: tuple[int, int, int]) -> str:
    """Map an RGB tuple to its color name, or 'unknown'."""
    return _RGB_TO_NAME.get(rgb, "unknown")


def _contrasting_text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick light or dark text color that contrasts with the background."""
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    if luminance > 128:
        # Dark text options for light backgrounds (includes red)
        return random.choice(  # noqa: S311
            [
                (0, 0, 0),
                (0, 32, 78),
                (93, 14, 65),
                (30, 30, 30),
                (206, 17, 38),
            ]
        )
    else:
        # Light text options for dark backgrounds (includes orange)
        return random.choice(  # noqa: S311
            [
                (255, 255, 255),
                (253, 185, 39),
                (230, 230, 230),
                (255, 103, 31),
            ]
        )


def _stroke_color(text_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick a thin outline color that contrasts with the text."""
    luminance = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


class JerseyNumberRenderer:
    """Renders synthetic jersey number images with broadcast degradation.

    Args:
        font_dir: Directory containing .ttf font files.
        image_size: Output image size (square).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        font_dir: Path,
        image_size: int = 96,
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.rng = random.Random(seed)  # noqa: S311
        self.np_rng = np.random.default_rng(seed)

        # Load all TTF fonts from directory
        font_paths = sorted(font_dir.glob("*.ttf"))
        if not font_paths:
            raise FileNotFoundError(f"No .ttf fonts found in {font_dir}")
        self._font_paths = font_paths

        # Pre-build degradation transforms (always applied, p=1.0)
        self._jpeg = RandomJPEGCompression(quality_min=10, quality_max=60, p=1.0)
        self._downscale = RandomBilinearDownscale(scale_min=0.2, scale_max=0.6, p=1.0)
        self._noise = RandomGaussianNoise(sigma_min=5.0, sigma_max=30.0, p=1.0)

    def _random_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load a random font at the given pixel size."""
        path = self.rng.choice(self._font_paths)
        return ImageFont.truetype(str(path), size)

    def render(
        self,
        label: str,
        bg_color: tuple[int, int, int] | None = None,
        text_color: tuple[int, int, int] | None = None,
    ) -> tuple[Image.Image, dict[str, object]]:
        """Render a single synthetic jersey number image.

        Args:
            label: The digit string to render (e.g. "6", "46", "00").
            bg_color: Optional explicit background color (RGB). Random if None.
            text_color: Optional explicit text color (RGB). Random if None.

        Returns:
            Tuple of (PIL Image, metadata dict with jersey_color, number_color,
            border keys).
        """
        size = self.image_size

        # 1. Background: random jersey color + subtle noise for fabric texture
        if bg_color is None:
            bg_color = self.rng.choice(_JERSEY_BACKGROUNDS)
        img = Image.new("RGB", (size, size), bg_color)
        arr = np.array(img, dtype=np.float32)
        fabric_noise = self.np_rng.normal(0, 3.0, arr.shape).astype(np.float32)
        img = Image.fromarray(
            np.clip(arr + fabric_noise, 0, 255).astype(np.uint8), mode="RGB"
        )

        # 2. Render text
        if text_color is None:
            text_color = _contrasting_text_color(bg_color)
        outline_color = _stroke_color(text_color)

        # Font size: fill most of the image height
        font_size = self.rng.randint(int(size * 0.5), int(size * 0.8))
        font = self._random_font(font_size)

        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Center with slight random offset
        cx = (size - text_w) / 2 + self.rng.uniform(-size * 0.08, size * 0.08)
        cy = (size - text_h) / 2 + self.rng.uniform(-size * 0.08, size * 0.08)
        x = cx - bbox[0]
        y = cy - bbox[1]

        # Draw with thin stroke outline
        stroke_width = max(1, font_size // 30)
        draw.text(
            (x, y),
            label,
            fill=text_color,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=outline_color,
        )

        # 3. Random perspective warp + rotation
        img = self._apply_perspective(img)

        # 4. Broadcast degradation pipeline
        result: Image.Image = self._jpeg(img)
        result = self._downscale(result)
        result = self._noise(result)

        # Build metadata describing what was rendered
        metadata: dict[str, object] = {
            "jersey_color": _rgb_to_name(bg_color),
            "number_color": _rgb_to_name(text_color),
            "border": stroke_width > 0,
        }

        return result, metadata

    def _apply_perspective(self, img: Image.Image) -> Image.Image:
        """Apply random perspective warp and rotation."""
        w, h = img.size

        # Small random rotation
        angle = self.rng.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

        # Perspective warp: shift each corner slightly
        margin = w * 0.08
        src: list[tuple[float, float]] = [
            (0.0, 0.0),
            (float(w), 0.0),
            (float(w), float(h)),
            (0.0, float(h)),
        ]
        dst = [
            (self.rng.uniform(0, margin), self.rng.uniform(0, margin)),
            (w - self.rng.uniform(0, margin), self.rng.uniform(0, margin)),
            (w - self.rng.uniform(0, margin), h - self.rng.uniform(0, margin)),
            (self.rng.uniform(0, margin), h - self.rng.uniform(0, margin)),
        ]

        # Compute perspective transform coefficients
        coeffs = self._find_coeffs(dst, src)
        img = img.transform(
            (w, h),
            Image.Transform.PERSPECTIVE,
            coeffs,
            resample=Image.Resampling.BILINEAR,
        )

        return img

    @staticmethod
    def _find_coeffs(
        target: list[tuple[float, float]],
        source: list[tuple[float, float]],
    ) -> list[float]:
        """Compute perspective transform coefficients.

        Maps source quadrilateral to target quadrilateral.
        Returns 8 coefficients for PIL Image.transform(PERSPECTIVE).
        """
        matrix: list[list[float]] = []
        for s, t in zip(source, target, strict=True):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
        a = np.array(matrix, dtype=np.float64)
        b = np.array([c for p in source for c in p], dtype=np.float64)
        res: list[float] = np.linalg.solve(a, b).tolist()
        return res
