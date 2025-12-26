from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="X-Ray Bone Fracture Assessment (XAI)",
        version="0.1.0",
        description=(
            "Research-grade decision support tool for bone fracture assessment "
            "with feature-focused preprocessing and explainable AI (Grad-CAM)."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"] ,
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000.0:.2f}"
        return response

    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    return app


app = create_app()
