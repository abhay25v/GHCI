from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import shap

from ati_engine.api.schemas import ExplainResponse, TokenAttribution
from ati_engine.inference.service import InferenceService

logger = logging.getLogger(__name__)


class ShapExplainer:
    def __init__(self, service: InferenceService) -> None:
        self.service = service

    def _target_probability_fn(self, target_label: str):
        def f(inputs: List[str]) -> np.ndarray:
            probs = []
            for text in inputs:
                pred = self.service.predict(text, top_k=1)
                # Naively: 1.0 if top label matches, else 0.0 (since we don't expose full probability vector)
                # For better fidelity, extend service.model to return probability for a given candidate label.
                p = 1.0 if pred.primary_label == target_label else 0.0
                probs.append([p])
            return np.array(probs)

        return f

    def explain(self, text: str, target_label: str, max_tokens: int = 50) -> ExplainResponse:
        # SHAP text masker; keeps runtime bounded
        masker = shap.maskers.Text(tokenizer="auto")
        f = self._target_probability_fn(target_label)
        explainer = shap.Explainer(f, masker)
        shap_values = explainer([text])

        # Extract token attributions from first example
        attributions: List[TokenAttribution] = []
        try:
            tokens: List[str] = shap_values.data[0]
            values: List[float] = shap_values.values[0].tolist()
            pairs: List[Tuple[str, float]] = list(zip(tokens, values))
            # Sort by absolute importance and truncate
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            pairs = pairs[:max_tokens]
            attributions = [TokenAttribution(token=t, value=float(v)) for t, v in pairs]
        except Exception:
            logger.exception("Failed to extract SHAP token attributions")

        summary: Dict[str, Any] = {
            "num_tokens": len(attributions),
            "method": "shap_text_masker",
        }
        return ExplainResponse(input_text=text, target_label=target_label, attributions=attributions, summary=summary)
