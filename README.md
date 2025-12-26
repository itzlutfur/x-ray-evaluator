# Bone Fracture Assessment System (XAI)

Thesis alignment: **“Feature-Focused Preprocessing in Bone Fracture Imaging Using Explainable AI.”**

This project provides a research-oriented web system for:

- Uploading an X-ray image
- Validating that the input is X-ray-like (rejecting obvious non-X-ray / low-quality inputs)
- Running inference using **pre-trained Keras models** (no retraining)
- Generating **Grad-CAM** for clinically interpretable justification

**Disclaimer:** This system is a research-based decision support tool and not a replacement for professional medical diagnosis.

## Folder structure

- `backend/` FastAPI + TensorFlow/Keras inference, preprocessing, validation, Grad-CAM
- `frontend/` React UI (Vite)

## Models (required)

Place the provided `.keras` files here:

- `backend/models/DenseNet121.keras`
- `backend/models/DenseNet201.keras`
- `backend/models/ResNet50.keras`
- `backend/models/ResNet101.keras`
- `backend/models/MobileNetV2.keras`
- `backend/models/InceptionV3.keras`
- `backend/models/Xception.keras`

Or set `XRAY_MODEL_DIR` to point to the folder containing them.

## Backend: run (Windows PowerShell)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API:

- `GET http://localhost:8000/healthz`
- `GET http://localhost:8000/api/v1/inference/models`
- `POST http://localhost:8000/api/v1/inference/predict` (multipart form: `file`, `model_name`, optional `consent_store`)

Smoke test (runs without launching Uvicorn):

```powershell
cd backend
.\venv\Scripts\python.exe scripts\smoke_predict.py "C:\path\to\image.jpg" --model ResNet50 --summary
```

## Frontend: run

```powershell
cd frontend
npm install
npm run dev
```

If your backend runs elsewhere:

```powershell
setx VITE_API_BASE "http://localhost:8000"
```

To build production assets:

```powershell
npm run build
```

## Validation & testing

- Fracture sample (IMG0002446.jpg): predicts **Fracture**, emits Grad-CAM overlay and low-confidence warning.
- Non-fracture sample (15600000.jpg): predicts **Non-Fracture**, confidence ≈ 0.59, Grad-CAM overlay available.
- Non X-ray (61xya9.png cat photo): blocked by validator with natural-image warning.
- `scripts/smoke_predict.py --summary` outputs model, prediction, confidence, and Grad-CAM status for spot checks.

## Demo walkthrough

1. Start backend with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
2. Start frontend with `npm run dev -- --host` and open the served URL.
3. Upload fracture vs non-fracture samples, toggle Grad-CAM, review warnings/consent checkbox, and discuss validation rationale.

## Design choices (why this matches the thesis)

- **Feature-focused preprocessing**: centralized in `backend/app/utils/preprocessing.py` with **CLAHE + gamma correction** and **224×224** resizing.
- **Safety-first validation**: `backend/app/utils/validation.py` implements best-effort heuristics to reject corrupt/low-quality inputs, many natural photos, and CT-like slices.
- **Explainability as justification**: `backend/app/utils/gradcam.py` finds the **last convolutional layer dynamically** and computes Grad-CAM; failures are reported and the UI warns accordingly.
- **No default image storage**: images are not written to disk unless `consent_store=true` is provided.

## Notes

- This repo pins TensorFlow to **2.20.x** (works with Windows + Python 3.13). If you need CPU/GPU-specific guidance, tell me your setup and I’ll adjust pins accordingly.
