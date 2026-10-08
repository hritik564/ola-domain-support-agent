"""Small, explicit prompt-injection denylist for the deterministic demo."""

from __future__ import annotations

import re
from dataclasses import dataclass

_INJECTION_PATTERNS = (
    re.compile(
        r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior)\s+"
        r"(?:system\s+|developer\s+)?instructions?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|show|print|disclose)\s+(?:the\s+)?"
        r"(?:system|developer)\s+prompt\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:disable|bypass|ignore)\s+(?:the\s+)?"
        r"(?:safety\s+)?(?:guardrails|safety rules)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpretend\s+(?:the\s+)?tool\s+(?:said|returned|reported)\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class InjectionCheck:
    blocked: bool
    matched_rule: str | None = None


def check_prompt_injection(text: str) -> InjectionCheck:
    for index, pattern in enumerate(_INJECTION_PATTERNS, start=1):
        if pattern.search(text):
            return InjectionCheck(blocked=True, matched_rule=f"rule-{index}")
    return InjectionCheck(blocked=False)
