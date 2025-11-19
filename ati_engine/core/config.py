from __future__ import annotations

import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    APP_VERSION: str = Field("0.1.0", description="Application version")
    MODEL_NAME: str = Field(
        default=os.getenv("MODEL_NAME", "typeform/distilbert-base-uncased-mnli"),
        description="Hugging Face model to use for zero-shot classification (DistilBERT MNLI).",
    )
    DEVICE: int = Field(-1, description="Inference device id (-1 for CPU)")
    TAXONOMY_PATH: str = Field(
        default=os.getenv("TAXONOMY_PATH", "ati_engine/taxonomy/sample_taxonomy.yaml"),
        description="Path to YAML taxonomy file.",
    )
    MAX_CANDIDATES: int = Field(20, description="Maximum number of taxonomy labels to consider")
    SHAP_MAX_SAMPLES: int = Field(50, description="Maximum number of samples for SHAP background")
    LOG_LEVEL: str = Field("INFO", description="Logging level")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
