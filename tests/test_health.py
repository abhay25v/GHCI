from __future__ import annotations

from ati_engine.core.config import settings


def test_settings_loaded():
    assert isinstance(settings.APP_VERSION, str)
    assert settings.APP_VERSION
