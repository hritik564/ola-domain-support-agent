"""Lookup support-ticket status and calculate a dataset-aware urgency score."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from dataset import SUPPORT_TICKETS

MAX_TICKET_AGE_DAYS = 30
ESCALATED_FLAG_WEIGHT = 0.65
AGE_SIGNAL_WEIGHT = 0.35
RECOMMENDATION_PERCENTILE = 0.80
RECOMMENDATION_ELIGIBLE_STATUSES = ("Open", "In Progress")


def calculate_escalation_score(escalated: bool, days_since_created: int) -> float:
    """Combine current escalation state with normalized ticket age."""
    if not isinstance(escalated, bool):
        raise ValueError("escalated must be a boolean.")
    if (
        not isinstance(days_since_created, int)
        or isinstance(days_since_created, bool)
        or not 0 <= days_since_created <= MAX_TICKET_AGE_DAYS
    ):
        raise ValueError("days_since_created must be an integer from 0 to 30.")

    age_signal = days_since_created / MAX_TICKET_AGE_DAYS
    score = (
        ESCALATED_FLAG_WEIGHT * int(escalated)
        + AGE_SIGNAL_WEIGHT * age_signal
    )
    return min(1.0, max(0.0, score))


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    """Compute a percentile as sorted[ceil(p*n)-1], using one-based rank."""
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty sequence.")
    if not isinstance(percentile, (int, float)) or not 0 < percentile <= 1:
        raise ValueError("percentile must be greater than 0 and at most 1.")
    ordered = sorted(values)
    rank = math.ceil(float(percentile) * len(ordered))
    return float(ordered[rank - 1])


def get_escalation_recommendation_threshold(
    tickets: Sequence[dict[str, Any]] | None = None,
) -> float:
    """Use the dataset's P80 age contribution as the escalation cutoff."""
    records = list(SUPPORT_TICKETS if tickets is None else tickets)
    if not records:
        raise ValueError("Cannot derive an escalation threshold from no tickets.")
    p80_age_days = nearest_rank_percentile(
        [ticket["days_since_created"] for ticket in records],
        RECOMMENDATION_PERCENTILE,
    )
    return AGE_SIGNAL_WEIGHT * (p80_age_days / MAX_TICKET_AGE_DAYS)


def check_support_ticket_status(record_id: str) -> dict[str, Any]:
    """Return a deterministic ticket summary or raise for an unknown ID."""
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a non-empty string.")

    ticket = next(
        (item for item in SUPPORT_TICKETS if item["record_id"] == record_id),
        None,
    )
    if ticket is None:
        raise ValueError(f"Support ticket not found: {record_id}")

    threshold = get_escalation_recommendation_threshold()
    age_signal = ticket["days_since_created"] / MAX_TICKET_AGE_DAYS
    score = calculate_escalation_score(
        ticket["escalated"],
        ticket["days_since_created"],
    )
    return {
        "record_id": ticket["record_id"],
        "status": ticket["status"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "escalated": ticket["escalated"],
        "days_since_created": ticket["days_since_created"],
        "age_signal": age_signal,
        "escalation_score": score,
        "escalation_recommended": (
            ticket["status"] in RECOMMENDATION_ELIGIBLE_STATUSES
            and score >= threshold
        ),
        "recommendation_threshold": threshold,
    }
