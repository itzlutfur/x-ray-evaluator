# System Summary

## Backend (FastAPI)

- `app/main.py`
  - Configures FastAPI app, timing middleware, CORS, mounts API router.
- `app/api/routes/inference.py`
  - `GET /api/v1/inference/models`: lists cached `.keras` models from `model_registry`.
  - `POST /api/v1/inference/predict`: handles upload, validation, inference, Grad-CAM PNG response, consent flag.
- `app/services/model_registry.py`
  - Loads TensorFlow models once and caches them; supports seven provided backbones.
- `app/services/inference_service.py`
  - Orchestrates validation (`validation.validate_xray_like_image`), thesis preprocessing, single-batch prediction, Grad-CAM, warnings, consent storage.
  - Fix: interprets single-neuron model outputs as non-fracture probability (inverted for fracture probability) and reports `probability_fracture` + `confidence`.
- `app/utils`
  - `image_io.py`: Pillow decode with EXIF rotation; returns RGB ndarray.
  - `preprocessing.py`: thesis pipeline (grayscale → CLAHE → gamma → resize), now outputs 0–255 float32 for model’s internal `Rescaling(1./255)`.
  - `validation.py`: adaptive heuristics (contrast, blur, color, CT-like circle, edge density) to accept bone X-rays yet reject natural photos.
  - `gradcam.py`: auto-detects last conv layer even in nested backbones; returns heatmap/overlay PNGs with fallbacks.
- `scripts/smoke_predict.py`: FastAPI TestClient smoke test; `--summary` prints concise JSON (model, label, confidence, Grad-CAM status/length).

## Frontend (React + Vite)

- `src/App.tsx`: Upload UI (drag/drop), model selector, consent checkbox, prediction card, confidence/warnings, Grad-CAM toggle, validation reason list, disclaimer.
- `src/lib/api.ts`: Fetch model list & run prediction; converts base64 PNGs for preview.
- Built with `npm run build` (verified).

## Validation & Testing

- Backend verifies on:
  - Fracture: `IMG0002446.jpg` — prediction “Fracture”, Grad-CAM OK, low-confidence warning.
  - Non-Fracture: `15600000.jpg` — prediction “Non-Fracture”, confidence ~0.59.
  - Non X-ray (cat): `C:\Users\tanvi\Downloads\61xya9.png` — rejected by validator as natural photo.

## Key Fixes Delivered

1. Grad-CAM robustness across all `.keras` backbones (layer detection, backbone reconnection, PNG generation).
2. Validation tuning (color/contrast thresholds, adaptive edge density) balancing safety versus acceptable clinical images.
3. Correct inference interpretation after discovering models output non-fracture probability; matched expected preprocessing scale.

## Demo Checklist

1. Start backend:
   ```powershell
   cd backend
   .\venv\Scripts\activate
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
2. Start frontend (new terminal):
   ```powershell
   cd frontend
   npm run dev -- --host
   ```
3. Open Vite URL (`http://localhost:5173`), upload fracture vs non-fracture samples, toggle Grad-CAM, show warnings/disclaimer.
