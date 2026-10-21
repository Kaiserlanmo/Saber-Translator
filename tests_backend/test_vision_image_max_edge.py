"""Tests for the configurable long-edge limit on vision images."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.image_helpers import encode_vision_image


def _decode(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_landscape_image_is_scaled_to_the_configured_long_edge() -> None:
    with Image.new("RGB", (850, 500), "white") as image:
        encoded, media_type = encode_vision_image(
            image,
            compress=True,
            max_edge=400,
        )
    assert media_type == "image/jpeg"
    with _decode(encoded) as decoded:
        assert decoded.size == (400, 235)


def test_portrait_image_keeps_its_aspect_ratio() -> None:
    with Image.new("RGB", (500, 850), "white") as image:
        encoded, _ = encode_vision_image(image, compress=True, max_edge=400)
    with _decode(encoded) as decoded:
        assert decoded.size == (235, 400)


def test_image_smaller_than_the_limit_is_not_upscaled() -> None:
    with Image.new("RGB", (300, 200), "white") as image:
        encoded, _ = encode_vision_image(image, compress=True, max_edge=2048)
    with _decode(encoded) as decoded:
        assert decoded.size == (300, 200)


def test_zero_means_no_limit() -> None:
    with Image.new("RGB", (850, 500), "white") as image:
        encoded, _ = encode_vision_image(image, compress=True, max_edge=0)
    with _decode(encoded) as decoded:
        assert decoded.size == (850, 500)


def test_lossless_png_is_not_resized() -> None:
    with Image.new("RGB", (850, 500), "white") as image:
        encoded, media_type = encode_vision_image(
            image,
            compress=False,
            max_edge=400,
        )
    assert media_type == "image/png"
    with _decode(encoded) as decoded:
        assert decoded.size == (850, 500)


def test_invalid_limit_is_rejected() -> None:
    with Image.new("RGB", (32, 32), "white") as image:
        for invalid in (-1, 16385, True, "2048"):
            with pytest.raises(ValueError):
                encode_vision_image(image, compress=True, max_edge=invalid)
