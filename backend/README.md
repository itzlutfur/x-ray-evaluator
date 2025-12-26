# Backend (FastAPI)

## Models

Place your provided `.keras` files under `backend/models/`:

- DenseNet121.keras
- DenseNet201.keras
- ResNet50.keras
- ResNet101.keras
- MobileNetV2.keras
- InceptionV3.keras
- Xception.keras

Or set `XRAY_MODEL_DIR` to a custom folder.

## Run

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Note: On Windows with Python 3.13, TensorFlow wheels are available for TF 2.20.x, so `requirements.txt` is pinned accordingly.

API:

- `GET /healthz`
- `GET /api/v1/inference/models`
- `POST /api/v1/inference/predict`

## Smoke test (no server)

```bash
cd backend
.\venv\Scripts\python.exe scripts\smoke_predict.py "C:\\path\\to\\xray.jpg" --model ResNet50
```
