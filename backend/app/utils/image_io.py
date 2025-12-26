from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class DecodedImage:
    pil: Image.Image
    rgb: np.ndarray  # HxWx3 uint8


def decode_image_bytes(data: bytes) -> DecodedImage:
    try:
        with Image.open(io := _bytes_to_filelike(data)) as img:
            img = img.copy()
    except UnidentifiedImageError as e:
        raise ValueError("Uploaded file is not a valid image.") from e
    except Exception as e:
        raise ValueError("Unable to read the uploaded image.") from e

    img = _apply_exif_orientation(img)

    rgb_img = img.convert("RGB")
    rgb = np.array(rgb_img, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Unsupported image channels.")

    return DecodedImage(pil=rgb_img, rgb=rgb)


def _bytes_to_filelike(data: bytes):
    import io

    return io.BytesIO(data)


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    try:
        exif = img.getexif()
        if not exif:
            return img
        orientation = exif.get(274)
        if orientation == 3:
            return img.rotate(180, expand=True)
        if orientation == 6:
            return img.rotate(270, expand=True)
        if orientation == 8:
            return img.rotate(90, expand=True)
        return img
    except Exception:
        return img
