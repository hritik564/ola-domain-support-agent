"""Reject composer output that is not supported by upstream tool evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class GroundednessCheck:
    accepted: bool
    reason: str


def validate_policy_draft(
    draft: str,
    evidence: dict[str, Any],
) -> GroundednessCheck:
    expected = evidence.get("answer")
    if not isinstance(expected, str) or not expected.strip():
        return GroundednessCheck(False, "missing grounded-generation answer")
    if not evidence.get("grounded", False) and not evidence.get(
        "fallback_triggered", False
    ):
        return GroundednessCheck(False, "upstream policy result is not grounded")
    if _normalize(draft) != _normalize(expected):
        return GroundednessCheck(
            False,
            "composer changed or added content beyond the grounded-generation result",
        )
    return GroundednessCheck(True, "exact grounded-generation answer preserved")


def validate_ticket_draft(
    draft: dict[str, Any],
    ticket_evidence: dict[str, Any],
) -> GroundednessCheck:
    expected_fields = {
        "record_id": "record_id",
        "ticket_status": "status",
        "resolution_time_hours": "resolution_time_hours",
        "ticket_age_days": "days_since_created",
        "escalation_score": "escalation_score",
        "escalation_recommended": "escalation_recommended",
    }
    for response_field, evidence_field in expected_fields.items():
        if draft.get(response_field) != ticket_evidence.get(evidence_field):
            return GroundednessCheck(
                False,
                f"{response_field} does not match ticket lookup evidence",
            )
    return GroundednessCheck(True, "ticket fields match lookup evidence")
