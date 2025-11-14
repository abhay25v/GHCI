from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ati_engine.api.schemas import (
    InferenceRequest,
    InferenceResponse,
    Prediction,
    ExplainRequest,
    ExplainResponse,
    TaxonomyResponse,
)
from ati_engine.core.config import settings
from ati_engine.inference.service import InferenceService, get_inference_service
from ati_engine.taxonomy.loader import TaxonomyLoader
from ati_engine.xai.explainer import ShapExplainer

router = APIRouter(tags=["inference"]) 
logger = logging.getLogger(__name__)


@router.post("/infer", response_model=InferenceResponse)
async def infer(
    payload: InferenceRequest,
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    try:
        result = service.predict(payload.text, top_k=payload.top_k, include_scores=payload.include_scores)
        return result
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", response_model=ExplainResponse)
async def explain(
    payload: ExplainRequest,
    service: InferenceService = Depends(get_inference_service),
) -> ExplainResponse:
    try:
        top_pred = service.predict(payload.text, top_k=1)
        target_label = payload.target_label or top_pred.primary_label
        explainer = ShapExplainer(service)
        return explainer.explain(payload.text, target_label=target_label, max_tokens=payload.max_tokens)
    except Exception as e:
        logger.exception("Explain failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/taxonomy", response_model=TaxonomyResponse)
async def get_taxonomy() -> TaxonomyResponse:
    try:
        loader = TaxonomyLoader(settings.TAXONOMY_PATH)
        taxonomy = loader.load()
        labels = loader.list_labels(taxonomy)
        return TaxonomyResponse(labels=labels, taxonomy=taxonomy)
    except Exception as e:
        logger.exception("Failed to load taxonomy")
        raise HTTPException(status_code=500, detail=str(e))
