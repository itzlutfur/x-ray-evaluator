from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="XRAY_", case_sensitive=False)

    api_prefix: str = "/api/v1"

    model_dir: Path = Field(default=Path(__file__).resolve().parents[2] / "models")

    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    confidence_low_threshold: float = 0.60

    disclaimer: str = (
        "This system is a research-based decision support tool and not a replacement "
        "for professional medical diagnosis."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
