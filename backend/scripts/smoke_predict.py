from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Allow running as: python scripts/smoke_predict.py ...
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test: run /predict on a local image file")
    parser.add_argument("image", type=str, help="Path to an image file")
    parser.add_argument("--model", type=str, default="ResNet50", help="Model name (e.g., ResNet50)")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="If set, prints a short summary (avoids dumping base64 PNG payloads)",
    )
    parser.add_argument(
        "--consent-store",
        action="store_true",
        help="If set, sends consent_store=true (writes image under backend/data/consented)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    with image_path.open("rb") as f:
        data = f.read()

    client = TestClient(app)

    files = {"file": (image_path.name, data, "image/jpeg")}
    form = {
        "model_name": args.model,
        "consent_store": "true" if args.consent_store else "false",
    }

    resp = client.post("/api/v1/inference/predict", files=files, data=form)
    print("status:", resp.status_code)

    payload = resp.json()
    if not args.summary:
        print(payload)
        return 0

    warnings = payload.get("warnings") or []
    pred = payload.get("prediction")
    gradcam = payload.get("gradcam") or {}
    heatmap_b64 = gradcam.get("heatmap_png_b64")
    overlay_b64 = gradcam.get("overlay_png_b64")

    if isinstance(pred, dict):
        label = pred.get("label")
        confidence = pred.get("confidence")
    else:
        label = pred
        confidence = payload.get("confidence")

    print(
        {
            "model": payload.get("model"),
            "label": label,
            "confidence": confidence,
            "warnings": warnings,
            "gradcam_status": gradcam.get("status"),
            "gradcam_message": gradcam.get("message"),
            "heatmap_b64_len": len(heatmap_b64) if isinstance(heatmap_b64, str) else None,
            "overlay_b64_len": len(overlay_b64) if isinstance(overlay_b64, str) else None,
        }
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
