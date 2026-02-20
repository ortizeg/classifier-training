"""Tests for custom degradation transforms."""

from __future__ import annotations

import random

import pytest
from PIL import Image

from classifier_training.transforms import (
    RandomBilinearDownscale,
    RandomGaussianNoise,
    RandomJPEGCompression,
    RandomPixelate,
)


@pytest.fixture()
def rgb_image() -> Image.Image:
    """Create a 64x64 RGB test image with non-uniform content."""
    img = Image.new("RGB", (64, 64), color=(128, 64, 200))
    # Add some variation so JPEG compression has something to degrade
    for x in range(0, 64, 8):
        for y in range(0, 64, 8):
            img.putpixel((x, y), (255, 0, 0))
    return img


# --- RandomJPEGCompression ---


class TestRandomJPEGCompression:
    def test_output_is_pil_image(self, rgb_image: Image.Image) -> None:
        t = RandomJPEGCompression(p=1.0)
        result = t(rgb_image)
        assert isinstance(result, Image.Image)

    def test_preserves_size(self, rgb_image: Image.Image) -> None:
        t = RandomJPEGCompression(p=1.0)
        result = t(rgb_image)
        assert result.size == rgb_image.size

    def test_preserves_mode(self, rgb_image: Image.Image) -> None:
        t = RandomJPEGCompression(p=1.0)
        result = t(rgb_image)
        assert result.mode == rgb_image.mode

    def test_p_zero_skips(self, rgb_image: Image.Image) -> None:
        t = RandomJPEGCompression(p=0.0)
        result = t(rgb_image)
        assert result is rgb_image

    def test_p_one_always_applies(self, rgb_image: Image.Image) -> None:
        t = RandomJPEGCompression(quality_min=1, quality_max=5, p=1.0)
        result = t(rgb_image)
        # Low quality JPEG should differ from original
        assert result is not rgb_image

    def test_invalid_quality_range(self) -> None:
        with pytest.raises(ValueError, match="quality_min"):
            RandomJPEGCompression(quality_min=80, quality_max=50)

    def test_invalid_quality_bounds(self) -> None:
        with pytest.raises(ValueError, match="quality_min"):
            RandomJPEGCompression(quality_min=0, quality_max=50)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError, match="p must be"):
            RandomJPEGCompression(p=1.5)

    def test_rejects_non_pil(self) -> None:
        t = RandomJPEGCompression(p=1.0)
        with pytest.raises(TypeError, match="PIL Image"):
            t("not_an_image")

    def test_deterministic_with_seed(self, rgb_image: Image.Image) -> None:
        t = RandomJPEGCompression(quality_min=20, quality_max=75, p=1.0)
        random.seed(42)
        r1 = t(rgb_image)
        random.seed(42)
        r2 = t(rgb_image)
        assert list(r1.tobytes()) == list(r2.tobytes())


# --- RandomPixelate ---


class TestRandomPixelate:
    def test_output_is_pil_image(self, rgb_image: Image.Image) -> None:
        t = RandomPixelate(p=1.0)
        result = t(rgb_image)
        assert isinstance(result, Image.Image)

    def test_preserves_size(self, rgb_image: Image.Image) -> None:
        t = RandomPixelate(p=1.0)
        result = t(rgb_image)
        assert result.size == rgb_image.size

    def test_preserves_mode(self, rgb_image: Image.Image) -> None:
        t = RandomPixelate(p=1.0)
        result = t(rgb_image)
        assert result.mode == rgb_image.mode

    def test_p_zero_skips(self, rgb_image: Image.Image) -> None:
        t = RandomPixelate(p=0.0)
        result = t(rgb_image)
        assert result is rgb_image

    def test_p_one_always_applies(self, rgb_image: Image.Image) -> None:
        t = RandomPixelate(scale_min=0.25, scale_max=0.25, p=1.0)
        result = t(rgb_image)
        assert result is not rgb_image

    def test_invalid_scale_range(self) -> None:
        with pytest.raises(ValueError, match="scale_min"):
            RandomPixelate(scale_min=0.8, scale_max=0.3)

    def test_invalid_scale_bounds(self) -> None:
        with pytest.raises(ValueError, match="scale_min"):
            RandomPixelate(scale_min=0.0, scale_max=0.5)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError, match="p must be"):
            RandomPixelate(p=-0.1)

    def test_rejects_non_pil(self) -> None:
        t = RandomPixelate(p=1.0)
        with pytest.raises(TypeError, match="PIL Image"):
            t("not_an_image")

    def test_non_square_image(self) -> None:
        img = Image.new("RGB", (100, 50))
        t = RandomPixelate(p=1.0)
        result = t(img)
        assert result.size == (100, 50)

    def test_deterministic_with_seed(self, rgb_image: Image.Image) -> None:
        t = RandomPixelate(scale_min=0.25, scale_max=0.75, p=1.0)
        random.seed(42)
        r1 = t(rgb_image)
        random.seed(42)
        r2 = t(rgb_image)
        assert list(r1.tobytes()) == list(r2.tobytes())


# --- RandomGaussianNoise ---


class TestRandomGaussianNoise:
    def test_output_is_pil_image(self, rgb_image: Image.Image) -> None:
        t = RandomGaussianNoise(p=1.0)
        result = t(rgb_image)
        assert isinstance(result, Image.Image)

    def test_preserves_size(self, rgb_image: Image.Image) -> None:
        t = RandomGaussianNoise(p=1.0)
        result = t(rgb_image)
        assert result.size == rgb_image.size

    def test_preserves_mode(self, rgb_image: Image.Image) -> None:
        t = RandomGaussianNoise(p=1.0)
        result = t(rgb_image)
        assert result.mode == rgb_image.mode

    def test_p_zero_skips(self, rgb_image: Image.Image) -> None:
        t = RandomGaussianNoise(p=0.0)
        result = t(rgb_image)
        assert result is rgb_image

    def test_p_one_always_applies(self, rgb_image: Image.Image) -> None:
        t = RandomGaussianNoise(sigma_min=20.0, sigma_max=25.0, p=1.0)
        result = t(rgb_image)
        assert result is not rgb_image

    def test_invalid_sigma_range(self) -> None:
        with pytest.raises(ValueError, match="sigma_min"):
            RandomGaussianNoise(sigma_min=30.0, sigma_max=10.0)

    def test_invalid_sigma_bounds(self) -> None:
        with pytest.raises(ValueError, match="sigma_min"):
            RandomGaussianNoise(sigma_min=0.0, sigma_max=10.0)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError, match="p must be"):
            RandomGaussianNoise(p=2.0)

    def test_rejects_non_pil(self) -> None:
        t = RandomGaussianNoise(p=1.0)
        with pytest.raises(TypeError, match="PIL Image"):
            t("not_an_image")

    def test_pixel_values_in_range(self, rgb_image: Image.Image) -> None:
        import numpy as np

        t = RandomGaussianNoise(sigma_min=25.0, sigma_max=25.0, p=1.0)
        result = t(rgb_image)
        arr = np.array(result)
        assert arr.min() >= 0
        assert arr.max() <= 255


# --- RandomBilinearDownscale ---


class TestRandomBilinearDownscale:
    def test_output_is_pil_image(self, rgb_image: Image.Image) -> None:
        t = RandomBilinearDownscale(p=1.0)
        result = t(rgb_image)
        assert isinstance(result, Image.Image)

    def test_preserves_size(self, rgb_image: Image.Image) -> None:
        t = RandomBilinearDownscale(p=1.0)
        result = t(rgb_image)
        assert result.size == rgb_image.size

    def test_preserves_mode(self, rgb_image: Image.Image) -> None:
        t = RandomBilinearDownscale(p=1.0)
        result = t(rgb_image)
        assert result.mode == rgb_image.mode

    def test_p_zero_skips(self, rgb_image: Image.Image) -> None:
        t = RandomBilinearDownscale(p=0.0)
        result = t(rgb_image)
        assert result is rgb_image

    def test_p_one_always_applies(self, rgb_image: Image.Image) -> None:
        t = RandomBilinearDownscale(scale_min=0.3, scale_max=0.3, p=1.0)
        result = t(rgb_image)
        assert result is not rgb_image

    def test_invalid_scale_range(self) -> None:
        with pytest.raises(ValueError, match="scale_min"):
            RandomBilinearDownscale(scale_min=0.8, scale_max=0.3)

    def test_invalid_scale_bounds(self) -> None:
        with pytest.raises(ValueError, match="scale_min"):
            RandomBilinearDownscale(scale_min=0.0, scale_max=0.5)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError, match="p must be"):
            RandomBilinearDownscale(p=-0.1)

    def test_rejects_non_pil(self) -> None:
        t = RandomBilinearDownscale(p=1.0)
        with pytest.raises(TypeError, match="PIL Image"):
            t("not_an_image")

    def test_non_square_image(self) -> None:
        img = Image.new("RGB", (100, 50))
        t = RandomBilinearDownscale(p=1.0)
        result = t(img)
        assert result.size == (100, 50)

    def test_deterministic_with_seed(self, rgb_image: Image.Image) -> None:
        t = RandomBilinearDownscale(scale_min=0.3, scale_max=0.75, p=1.0)
        random.seed(42)
        r1 = t(rgb_image)
        random.seed(42)
        r2 = t(rgb_image)
        assert list(r1.tobytes()) == list(r2.tobytes())
