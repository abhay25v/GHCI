from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ati_engine.api.schemas import InferenceResponse, Prediction
from ati_engine.core.config import settings
from ati_engine.preprocessing.cleaner import normalize_text
from ati_engine.inference.model import DistilBertClassifier
from ati_engine.taxonomy.loader import TaxonomyLoader

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self, model: Optional[DistilBertClassifier] = None, taxonomy_loader: Optional[TaxonomyLoader] = None) -> None:
        self.model = model or DistilBertClassifier()
        self.taxonomy_loader = taxonomy_loader or TaxonomyLoader(settings.TAXONOMY_PATH)
        self._taxonomy = None
        self._labels: Optional[List[str]] = None

    def _get_labels(self) -> List[str]:
        if self._labels is None:
            self._taxonomy = self.taxonomy_loader.load()
            self._labels = self.taxonomy_loader.list_labels(self._taxonomy)[: settings.MAX_CANDIDATES]
            if not self._labels:
                raise ValueError("No labels found in taxonomy")
        return self._labels

    def predict(self, text: str, top_k: int = 5, include_scores: bool = True) -> InferenceResponse:
        clean_text = normalize_text(text)
        labels = self._get_labels()
        raw = self.model.predict(clean_text, candidate_labels=labels, top_k=top_k, multi_label=False)
        predictions: List[Prediction] = [
            Prediction(label=lbl, score=float(scr)) for lbl, scr in zip(raw.get("labels", []), raw.get("scores", []))
        ]
        primary_label = predictions[0].label if predictions else "UNKNOWN"
        metadata: Dict[str, object] = {
            "model": self.model.model_name,
            "device": settings.DEVICE,
            "num_labels": len(labels),
        }
        return InferenceResponse(
            input_text=text,
            top_predictions=predictions[:top_k],
            primary_label=primary_label,
            metadata=metadata,
        )


# Dependency provider
_service_singleton: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = InferenceService()
    return _service_singleton
