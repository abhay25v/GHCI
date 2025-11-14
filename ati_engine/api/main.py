from __future__ import annotations

import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ati_engine.api.schemas import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ExplainRequest,
    ExplainResponse,
    TaxonomyResponse,
)
from ati_engine.api.routers import health as health_router
from ati_engine.api.routers import inference as inference_router
from ati_engine.core.config import settings
from ati_engine.core.logging import configure_logging


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Autonomous Transaction Intelligence (ATI) Engine",
    version=settings.APP_VERSION,
    description="FastAPI microservice for DistilBERT inference, SHAP explainability, and YAML-based taxonomy mapping.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routers
app.include_router(health_router.router, prefix="/health")
app.include_router(inference_router.router, prefix="/v1")


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.APP_VERSION)
