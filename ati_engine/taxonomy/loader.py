from __future__ import annotations

import logging
from typing import Dict, List, Any
import yaml

logger = logging.getLogger(__name__)


class TaxonomyLoader:
    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("Taxonomy YAML must define a mapping at the root")
            return data
        except FileNotFoundError:
            logger.error("Taxonomy file not found: %s", self.path)
            raise

    def list_labels(self, taxonomy: Dict[str, Any]) -> List[str]:
        labels: List[str] = []
        for cat, spec in taxonomy.items():
            labels.append(cat)
            # Optionally include sub-categories as additional candidates
            subs = spec.get("subcategories") if isinstance(spec, dict) else None
            if isinstance(subs, list):
                labels.extend([f"{cat}::{sub}" for sub in subs])
        return labels
