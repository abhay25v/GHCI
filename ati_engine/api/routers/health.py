from __future__ import annotations

from fastapi import APIRouter

from ati_engine.api.schemas import HealthResponse
from ati_engine.core.config import settings

router = APIRouter()


@router.get("/z", response_model=HealthResponse, tags=["health"])
def healthz() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.APP_VERSION)
