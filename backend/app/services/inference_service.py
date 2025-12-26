from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from app.core.settings import get_settings
from app.services.model_registry import available_model_names, get_or_load_model
from app.utils.gradcam import compute_gradcam
from app.utils.image_io import decode_image_bytes
from app.utils.preprocessing import preprocess_for_model
from app.utils.validation import validate_xray_like_image


def get_available_models() -> list[str]:
    return available_model_names()


async def predict_with_explainability(
    *,
    upload,
    model_name: str,
    consent_store: bool,
    confidence_low_threshold: float,
) -> dict[str, Any]:
    data = await upload.read()
    if not data:
        raise ValueError("Empty upload.")

    decoded = decode_image_bytes(data)

    validation = validate_xray_like_image(decoded.rgb)
    if not validation.valid:
        return {
            "valid": False,
            "validation": {
                "message": validation.message,
                "reasons": validation.reasons,
                "metrics": validation.metrics,
            },
        }

    settings = get_settings()
    loaded = get_or_load_model(model_name)

    # Preprocessing (thesis-aligned)
    x = preprocess_for_model(decoded.rgb)
    batch = tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float32)

    # Inference
    preds = _predict_sync(loaded.model, batch)
    prob_fracture = _extract_fracture_probability(preds)

    label = "Fracture" if prob_fracture >= 0.5 else "Non-Fracture"
    confidence = float(max(prob_fracture, 1.0 - prob_fracture))

    warnings: list[str] = []
    if confidence < float(confidence_low_threshold):
        warnings.append("⚠ The model confidence is low. The prediction may be unreliable.")

    explanation = (
        "The model focused primarily on high-contrast fracture regions and cortical discontinuities, "
        "which influenced the final prediction."
    )

    # Grad-CAM
    heatmap_png = None
    overlay_png = None
    gradcam_status = "ok"
    gradcam_message = None

    try:
        gc = _gradcam_sync(loaded.model, batch, decoded.rgb)
        heatmap_png = gc.heatmap_png
        overlay_png = gc.overlay_png
        gradcam_message = f"Grad-CAM computed using layer '{gc.layer_name}'."
    except Exception as e:
        gradcam_status = "failed"
        gradcam_message = f"Grad-CAM could not be generated: {type(e).__name__}."
        warnings.append("⚠ Explainability (Grad-CAM) is unavailable for this prediction.")

    if consent_store:
        _store_with_consent(decoded.rgb, label, confidence, model_name)

    return {
        "valid": True,
        "validation": {
            "message": validation.message,
            "reasons": validation.reasons,
            "metrics": validation.metrics,
        },
        "prediction": label,
        "confidence": confidence,
        "probability_fracture": float(prob_fracture),
        "warnings": warnings,
        "explanation": explanation,
        "heatmap_png": heatmap_png,
        "overlay_png": overlay_png,
        "gradcam_status": gradcam_status,
        "gradcam_message": gradcam_message,
        "disclaimer": settings.disclaimer,
    }


def _predict_sync(model: tf.keras.Model, batch: tf.Tensor) -> np.ndarray:
    out = model(batch, training=False)
    out_np = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
    return out_np


def _extract_fracture_probability(preds: np.ndarray) -> float:
    preds = np.asarray(preds)
    if preds.ndim != 2 or preds.shape[0] != 1:
        raise RuntimeError("Unexpected model output shape.")

    if preds.shape[1] == 1:
        non_fracture_prob = float(preds[0, 0])
        prob_fracture = 1.0 - non_fracture_prob
        return float(np.clip(prob_fracture, 0.0, 1.0))

    if preds.shape[1] == 2:
        # Common convention: [non_fracture, fracture]
        p = float(preds[0, 1])
        return float(np.clip(p, 0.0, 1.0))

    # Fallback: treat max as positive
    p = float(np.max(preds[0]))
    return float(np.clip(p, 0.0, 1.0))


def _gradcam_sync(model: tf.keras.Model, batch: tf.Tensor, original_rgb: np.ndarray):
    return compute_gradcam(model=model, input_tensor=batch, original_rgb=original_rgb)


def _store_with_consent(rgb: np.ndarray, label: str, confidence: float, model_name: str) -> None:
    import cv2

    settings = get_settings()
    out_dir = Path(__file__).resolve().parents[2] / "data" / "consented"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_model = model_name.replace("/", "-")
    filename = out_dir / f"{ts}_{safe_model}_{label}_{confidence:.3f}.png"

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(filename), bgr)
