"""
图像处理辅助模块，提供图像处理相关的通用函数
"""

import base64
import io
import logging

from PIL import Image


VISION_IMAGE_JPEG_QUALITY = 95
# 视觉模型图片长边上限的合法范围。0 表示不缩放（保持旧行为）。
VISION_IMAGE_MAX_EDGE_MIN = 0
VISION_IMAGE_MAX_EDGE_MAX = 16384
VISION_IMAGE_MAX_EDGE_DEFAULT = 2048

logger = logging.getLogger("ImageHelpers")


def encode_vision_image(
    image: Image.Image,
    *,
    compress: bool,
    max_edge: int | None = None,
) -> tuple[str, str]:
    """Encode a PIL image for an OpenAI-compatible vision request.

    When ``compress`` is true, the image is converted to JPEG.  If ``max_edge``
    is a positive integer, the image is first scaled down proportionally so
    that its longer edge does not exceed that value.  Scaling never enlarges
    an image and never changes its aspect ratio.  The resized image exists only
    in memory and is released before this function returns, so no cache or
    temporary file is left behind.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("待编码图片必须是 PIL Image")
    if not isinstance(compress, bool):
        raise TypeError("视觉模型图片压缩开关必须是布尔值")
    if max_edge is not None and (
        isinstance(max_edge, bool)
        or not isinstance(max_edge, int)
        or max_edge < VISION_IMAGE_MAX_EDGE_MIN
        or max_edge > VISION_IMAGE_MAX_EDGE_MAX
    ):
        raise ValueError(
            "视觉模型图片长边上限必须是 "
            f"{VISION_IMAGE_MAX_EDGE_MIN} 到 {VISION_IMAGE_MAX_EDGE_MAX} 的整数"
        )

    source = image
    resized = None
    if compress and max_edge and max(image.size) > max_edge:
        width, height = image.size
        try:
            if width >= height:
                target = (
                    max_edge,
                    max(1, round(height * max_edge / width)),
                )
            else:
                target = (
                    max(1, round(width * max_edge / height)),
                    max_edge,
                )
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            resized = image.resize(target, resampling)
        except MemoryError:
            logger.error(
                "视觉模型图片缩放内存不足: %sx%s -> 长边 %s",
                width,
                height,
                max_edge,
            )
            raise
        except Exception as exc:
            logger.error(
                "视觉模型图片缩放失败: %sx%s -> 长边 %s: %s",
                width,
                height,
                max_edge,
                exc,
            )
            raise
        source = resized
        logger.debug(
            "视觉模型图片等比缩放: %sx%s -> %sx%s (长边上限 %s)",
            width,
            height,
            resized.width,
            resized.height,
            max_edge,
        )

    converted = source.convert("RGB") if compress and source.mode != "RGB" else source
    try:
        with io.BytesIO() as buffered:
            if compress:
                converted.save(
                    buffered,
                    format="JPEG",
                    quality=VISION_IMAGE_JPEG_QUALITY,
                    optimize=True,
                    subsampling=0,
                )
                media_type = "image/jpeg"
            else:
                converted.save(buffered, format="PNG")
                media_type = "image/png"
            encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
    finally:
        if converted is not source and converted is not image:
            converted.close()
        if resized is not None and resized is not converted:
            resized.close()
    return encoded, media_type


def image_to_rgb_array(image: Image.Image):
    """Copy a PIL image into a three-channel NumPy array."""

    if not isinstance(image, Image.Image):
        raise TypeError("待转换图片必须是 PIL Image")
    import numpy as np

    converted = image.convert("RGB")
    try:
        return np.array(converted)
    finally:
        if converted is not image:
            converted.close()
