from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from transformers import pipeline, Pipeline

from ati_engine.core.config import settings

logger = logging.getLogger(__name__)


class DistilBertClassifier:
    """Wrapper around Hugging Face pipelines to support zero-shot classification.

    Uses a DistilBERT model fine-tuned on MNLI for zero-shot classification where possible.
    Fallbacks to text-classification when the configured model doesn't support NLI.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[int] = None) -> None:
        self.model_name = model_name or settings.MODEL_NAME
        self.device = device if device is not None else settings.DEVICE
        self._zs_pipe: Optional[Pipeline] = None
        self._tc_pipe: Optional[Pipeline] = None

    def _get_zero_shot(self) -> Pipeline:
        if self._zs_pipe is None:
            logger.info("Loading zero-shot pipeline with model=%s device=%s", self.model_name, self.device)
            self._zs_pipe = pipeline(
                task="zero-shot-classification",
                model=self.model_name,
                device=self.device,
            )
        return self._zs_pipe

    def _get_text_class(self) -> Pipeline:
        if self._tc_pipe is None:
            fallback = "distilbert-base-uncased-finetuned-sst-2-english"
            logger.warning(
                "Falling back to text-classification with %s (no zero-shot available for %s)", fallback, self.model_name
            )
            self._tc_pipe = pipeline(task="text-classification", model=fallback, device=self.device)
        return self._tc_pipe

    def predict(
        self,
        text: str,
        candidate_labels: Optional[List[str]] = None,
        top_k: int = 5,
        multi_label: bool = False,
    ) -> Dict[str, Any]:
        """Run prediction.

        If candidate_labels are provided, attempts zero-shot classification.
        Otherwise falls back to sentiment classification.
        """
        if candidate_labels:
            try:
                zs = self._get_zero_shot()
                result = zs(text, candidate_labels=candidate_labels, multi_label=multi_label)
                # Ensure top_k handling
                labels = result["labels"][:top_k]
                scores = result["scores"][:top_k]
                return {"labels": labels, "scores": scores}
            except Exception:
                logger.exception("Zero-shot classification failed; using text-classification fallback")
                # continue to fallback
        tc = self._get_text_class()
        result = tc(text)
        if isinstance(result, list) and result:
            result = result[0]
        return {"labels": [result["label"]], "scores": [float(result["score"])], "task": "text-classification"}
