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
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        # Try to pull tokenizer from zero-shot pipeline; fallback to text-classification; else None
        try:
            pipe = self.service.model._get_zero_shot()
            self._tokenizer = getattr(pipe, "tokenizer", None)
        except Exception:
            try:
                pipe = self.service.model._get_text_class()
                self._tokenizer = getattr(pipe, "tokenizer", None)
            except Exception:
                logger.exception("Failed to acquire tokenizer for SHAP; will use simple tokenizer")
                self._tokenizer = None
        return self._tokenizer

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
        # Try to get a tokenizer; fallback to simple tokenization
        tokenizer = self._get_tokenizer()
        masker = shap.maskers.Text(tokenizer=tokenizer if tokenizer else "simple")
        f = self._target_probability_fn(target_label)
        shap_values = None
        explainer = None
        shap_error: str | None = None
        try:
            explainer = shap.Explainer(f, masker)
            shap_values = explainer([text])
        except Exception as e:
            shap_error = str(e)
            logger.exception("SHAP explanation failed; returning empty attributions")

        # Extract token attributions from first example
        attributions: List[TokenAttribution] = []
        if shap_values is not None:
            try:
                tokens: List[str] = shap_values.data[0]
                values: List[float] = shap_values.values[0].tolist()
                pairs: List[Tuple[str, float]] = list(zip(tokens, values))
                pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                pairs = pairs[:max_tokens]
                attributions = [TokenAttribution(token=t, value=float(v)) for t, v in pairs]
            except Exception:
                logger.exception("Failed to extract SHAP token attributions")
        else:
            # Basic whitespace token fallback for transparency
            simple_tokens = text.split()
            for t in simple_tokens[: max_tokens]:
                attributions.append(TokenAttribution(token=t, value=0.0))

        summary: Dict[str, Any] = {
            "num_tokens": len(attributions),
            "method": "shap_text_masker",
            "fallback": shap_values is None,
            "error": shap_error,
        }
        return ExplainResponse(input_text=text, target_label=target_label, attributions=attributions, summary=summary)
