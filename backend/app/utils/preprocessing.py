from __future__ import annotations

import cv2
import numpy as np


def preprocess_for_model(rgb: np.ndarray, *, gamma: float = 1.2, clahe_clip: float = 2.0) -> np.ndarray:
    """Feature-focused preprocessing consistent with thesis methodology.

    Steps:
    - Convert to grayscale (X-rays are intensity images)
    - CLAHE to enhance local bone contrast
    - Gamma correction to improve visibility of cortical edges
    - Resize to 224x224 (training resolution)
    - Normalize to [0, 1]
    - Replicate channels to 3 for CNN backbones

    Returns: float32 array of shape (224, 224, 3)
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image array.")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Gamma correction (operate in [0,1])
    x = gray_clahe.astype(np.float32) / 255.0
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, 1.0 / float(gamma))

    # Resize to model input
    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)

    # Model's own preprocessing pipeline expects 0-255 input before rescaling.
    x3 = np.stack([x, x, x], axis=-1).astype(np.float32) * 255.0
    return x3
