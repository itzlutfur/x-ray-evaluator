from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.settings import get_settings
from app.services.inference_service import (
    get_available_models,
    predict_with_explainability,
)

router = APIRouter()


@router.get("/models")
async def list_models() -> dict[str, Any]:
    return {"models": get_available_models()}


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    consent_store: bool = Form(False),
) -> dict[str, Any]:
    settings = get_settings()

    if model_name not in get_available_models():
        raise HTTPException(status_code=400, detail="Unsupported model. Choose one from /models.")

    try:
        result = await predict_with_explainability(
            upload=file,
            model_name=model_name,
            consent_store=consent_store,
            confidence_low_threshold=settings.confidence_low_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    def b64_png(data: bytes | None) -> str | None:
        if data is None:
            return None
        return base64.b64encode(data).decode("utf-8")

    return {
        "model": model_name,
        "valid": result["valid"],
        "validation": result["validation"],
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "warnings": result.get("warnings", []),
        "explanation": result.get("explanation"),
        "gradcam": {
            "heatmap_png_b64": b64_png(result.get("heatmap_png")),
            "overlay_png_b64": b64_png(result.get("overlay_png")),
            "status": result.get("gradcam_status"),
            "message": result.get("gradcam_message"),
        },
        "disclaimer": settings.disclaimer,
    }
