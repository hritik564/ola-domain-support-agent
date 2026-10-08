"""Deterministic masking for supported Indian mobile-number formats."""

from __future__ import annotations

import re
from dataclasses import dataclass

_INDIAN_PHONE = re.compile(
    r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{4}[-\s]?\d{5}(?!\d)"
)
REDACTION = "[PHONE REDACTED]"


@dataclass(frozen=True)
class PIIMaskResult:
    text: str
    masked: bool


def mask_phone_numbers(text: str) -> PIIMaskResult:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    masked_text, replacements = _INDIAN_PHONE.subn(REDACTION, text)
    return PIIMaskResult(text=masked_text, masked=replacements > 0)
