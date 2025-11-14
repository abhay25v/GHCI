from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")


class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Transaction description or free text to classify")
    top_k: int = Field(5, ge=1, le=20, description="Number of top candidate labels to return")
    include_scores: bool = Field(True, description="Whether to return per-label scores")


class Prediction(BaseModel):
    label: str
    score: float


class InferenceResponse(BaseModel):
    input_text: str
    top_predictions: List[Prediction]
    primary_label: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target_label: Optional[str] = Field(None, description="Label to explain; defaults to the top predicted label")
    max_tokens: int = Field(50, ge=5, le=256, description="Maximum number of tokens to attribute")


class TokenAttribution(BaseModel):
    token: str
    value: float


class ExplainResponse(BaseModel):
    input_text: str
    target_label: str
    attributions: List[TokenAttribution]
    summary: Dict[str, Any] = Field(default_factory=dict)


class TaxonomyResponse(BaseModel):
    labels: List[str]
    taxonomy: Dict[str, Any]
