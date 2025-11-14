from __future__ import annotations

import re
import sys
import unicodedata
from typing import Iterable


_whitespace_re = re.compile(r"\s+")
_punct_tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))


def normalize_text(text: str) -> str:
    """Normalize input text for model ingestion.

    Steps:
    - Unicode NFKC normalization
    - Lowercasing
    - Collapse whitespace
    - Strip punctuation at ends
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    t = _whitespace_re.sub(" ", t).strip()
    return t


def strip_punctuation(text: str) -> str:
    return text.translate(_punct_tbl)
