from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    message: str
    reasons: list[str]
    metrics: dict[str, Any]


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_xray_like_image(rgb: np.ndarray) -> ValidationResult:
    """Heuristic validation for bone X-ray images.

    Note: Without a dedicated modality classifier, this is a best-effort, safety-minded
    heuristic gate to reject clearly invalid inputs (natural photos, extremely low quality,
    typical CT slice appearance).
    """

    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return ValidationResult(False, _msg_invalid(), ["Unsupported channel layout"], metrics)

    h, w, _ = rgb.shape
    metrics["height"] = int(h)
    metrics["width"] = int(w)

    if min(h, w) < 160:
        reasons.append("Image resolution is too low for reliable assessment")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Basic corruption / dynamic range checks
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))
    metrics["mean_intensity"] = mean_intensity
    metrics["std_intensity"] = std_intensity

    if std_intensity < 7.0:
        reasons.append("Image contrast is extremely low")

    # Blur check (Laplacian variance)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["laplacian_variance"] = lap_var
    if lap_var < 18.0:
        reasons.append("Image appears blurry or out of focus")

    # Colorfulness / saturation check to reject natural photos
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    metrics["saturation_mean"] = sat_mean

    # Also check how close channels are (X-rays are mostly grayscale)
    ch0 = rgb[:, :, 0].astype(np.int16)
    ch1 = rgb[:, :, 1].astype(np.int16)
    ch2 = rgb[:, :, 2].astype(np.int16)
    channel_diff = float(np.mean(np.abs(ch0 - ch1) + np.abs(ch1 - ch2) + np.abs(ch0 - ch2)) / 3.0)
    metrics["channel_diff_mean"] = channel_diff

    color_heavy = sat_mean > 24.0
    channel_misaligned = channel_diff > 8.0
    if color_heavy and channel_misaligned:
        reasons.append("Image appears to be a natural/color photograph rather than an X-ray")

    # CT-like slice detection heuristic: large central circle edge
    ct_like = _looks_like_ct_slice(gray)
    metrics["ct_like"] = bool(ct_like)
    if ct_like:
        reasons.append("Image resembles a CT/MRI slice rather than a projection X-ray")

    # Bone-like edge density heuristic
    edges = cv2.Canny(gray, 60, 180)
    edge_density = float(np.mean(edges > 0))
    metrics["edge_density"] = edge_density

    # Adapt the minimum structural requirement based on how grayscale/high-variance the image is.
    grayscale_like = sat_mean < 20.0 and channel_diff < 7.5
    min_edge_density = 0.004
    if grayscale_like:
        min_edge_density = 0.0025
    if lap_var > 45.0 or std_intensity > 30.0:
        min_edge_density *= 0.6

    metrics["edge_density_threshold"] = float(min_edge_density)

    if edge_density < min_edge_density:
        reasons.append("Image contains too little structural detail")
    if edge_density > 0.26:
        reasons.append("Image contains excessive texture/detail and may be non-X-ray")

    bright_ratio = float(np.mean(gray > 185))
    metrics["bright_ratio"] = bright_ratio

    valid = len(reasons) == 0
    return ValidationResult(
        valid=valid,
        message="✅ Valid bone X-ray detected." if valid else _msg_invalid(),
        reasons=reasons,
        metrics=metrics,
    )


def _looks_like_ct_slice(gray: np.ndarray) -> bool:
    # Downscale for speed
    h, w = gray.shape
    scale = 512.0 / max(h, w)
    if scale < 1.0:
        gray_small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray

    blur = cv2.GaussianBlur(gray_small, (5, 5), 1.0)
    edges = cv2.Canny(blur, 60, 160)

    # Hough circle detection to catch strong circular boundary typical in CT
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(gray_small.shape) / 3.0,
        param1=120,
        param2=35,
        minRadius=int(min(gray_small.shape) * 0.20),
        maxRadius=int(min(gray_small.shape) * 0.49),
    )

    if circles is None:
        return False

    circles = np.round(circles[0, :]).astype(int)
    # If there is a large circle near center, treat as CT-like
    cy0, cx0 = gray_small.shape[0] / 2.0, gray_small.shape[1] / 2.0
    for x, y, r in circles[:3]:
        dist = ((x - cx0) ** 2 + (y - cy0) ** 2) ** 0.5
        if dist < 0.12 * min(gray_small.shape) and r > 0.25 * min(gray_small.shape):
            # additionally require substantial edge pixels along circumference
            mask = np.zeros_like(edges, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, thickness=3)
            on_circle = edges[mask > 0]
            if float(np.mean(on_circle > 0)) > 0.08:
                return True
    return False


def _msg_invalid() -> str:
    return "❌ This image does not appear to be a valid bone X-ray. Please upload a clear X-ray image."
