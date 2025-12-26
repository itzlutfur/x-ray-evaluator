from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import tensorflow as tf

from app.core.settings import get_settings


@dataclass(frozen=True)
class LoadedModel:
    name: str
    path: Path
    model: tf.keras.Model


_MODEL_LOCK = Lock()
_MODELS: dict[str, LoadedModel] = {}


SUPPORTED_MODELS: dict[str, str] = {
    "DenseNet121": "DenseNet121.keras",
    "DenseNet201": "DenseNet201.keras",
    "ResNet50": "ResNet50.keras",
    "ResNet101": "ResNet101.keras",
    "MobileNetV2": "MobileNetV2.keras",
    "InceptionV3": "InceptionV3.keras",
    "Xception": "Xception.keras",
}


def available_model_names() -> list[str]:
    return sorted(SUPPORTED_MODELS.keys())


def get_or_load_model(name: str) -> LoadedModel:
    if name not in SUPPORTED_MODELS:
        raise ValueError("Unsupported model.")

    with _MODEL_LOCK:
        if name in _MODELS:
            return _MODELS[name]

        settings = get_settings()
        model_path = Path(settings.model_dir) / SUPPORTED_MODELS[name]
        if not model_path.exists():
            raise ValueError(
                f"Model file not found: {model_path}. Place the provided .keras files in the model directory."
            )

        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{name}'.") from e

        loaded = LoadedModel(name=name, path=model_path, model=model)
        _MODELS[name] = loaded
        return loaded
