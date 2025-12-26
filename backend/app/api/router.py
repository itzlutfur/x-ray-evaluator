from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import inference

api_router = APIRouter()
api_router.include_router(inference.router, prefix="/inference", tags=["inference"])
