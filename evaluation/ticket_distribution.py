"""Deterministic distribution analysis for generated support tickets."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from dataset import SUPPORT_TICKETS
from tools.ticket_lookup import (
    AGE_SIGNAL_WEIGHT,
    ESCALATED_FLAG_WEIGHT,
    MAX_TICKET_AGE_DAYS,
    RECOMMENDATION_ELIGIBLE_STATUSES,
    calculate_escalation_score,
    get_escalation_recommendation_threshold,
    nearest_rank_percentile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "ticket_distribution.json"
AGE_PERCENTILE = 0.80


def summarize_ticket_distribution(
    tickets: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = list(SUPPORT_TICKETS if tickets is None else tickets)
    if not records:
        raise ValueError("Cannot summarize an empty support-ticket dataset.")

    ages = [int(ticket["days_since_created"]) for ticket in records]
    scores = [
        calculate_escalation_score(
            ticket["escalated"],
            int(ticket["days_since_created"]),
        )
        for ticket in records
    ]
    age_p80 = nearest_rank_percentile(ages, AGE_PERCENTILE)
    threshold = get_escalation_recommendation_threshold(records)
    escalated_count = sum(ticket["escalated"] is True for ticket in records)
    eligible_count = sum(
        ticket["status"] in RECOMMENDATION_ELIGIBLE_STATUSES
        for ticket in records
    )
    recommended_count = sum(
        ticket["status"] in RECOMMENDATION_ELIGIBLE_STATUSES
        and score >= threshold
        for ticket, score in zip(records, scores)
    )
    score_at_or_above_threshold_count = sum(score >= threshold for score in scores)
    age_p80_rank = nearest_rank_index(len(ages), AGE_PERCENTILE)
    score_p80_rank = nearest_rank_index(len(scores), AGE_PERCENTILE)

    return {
        "ticket_count": len(records),
        "escalated_flag_counts": {
            "true": escalated_count,
            "false": len(records) - escalated_count,
        },
        "days_since_created": {
            "min": min(ages),
            "max": max(ages),
            "median": median(ages),
            "p80": age_p80,
            "percentile_method": "nearest_rank",
            "p80_rank_1_based": age_p80_rank + 1,
        },
        "escalation_score_distribution": {
            "min": min(scores),
            "max": max(scores),
            "median": median(scores),
            "p80": nearest_rank_percentile(scores, AGE_PERCENTILE),
            "p80_rank_1_based": score_p80_rank + 1,
            "recommended_count": recommended_count,
            "eligible_ticket_count": eligible_count,
            "not_recommended_eligible_count": eligible_count - recommended_count,
            "score_at_or_above_threshold_count": (
                score_at_or_above_threshold_count
            ),
            "ineligible_status_count": len(records) - eligible_count,
        },
        "score_formula": (
            "0.65 * int(escalated) + 0.35 * "
            "(days_since_created / 30)"
        ),
        "weights": {
            "escalated_flag": ESCALATED_FLAG_WEIGHT,
            "normalized_age": AGE_SIGNAL_WEIGHT,
            "sum": ESCALATED_FLAG_WEIGHT + AGE_SIGNAL_WEIGHT,
        },
        "age_normalization": "days_since_created / 30",
        "recommendation_threshold": threshold,
        "recommendation_threshold_derivation": {
            "age_p80_days": age_p80,
            "normalized_age_signal": age_p80 / MAX_TICKET_AGE_DAYS,
            "age_weight": AGE_SIGNAL_WEIGHT,
            "expression": (
                f"{AGE_SIGNAL_WEIGHT} * "
                f"({age_p80:g} / {MAX_TICKET_AGE_DAYS})"
            ),
        },
        "recommendation_rule": (
            "For Open and In Progress tickets, recommend escalation when the "
            "score reaches the age-weighted nearest-rank 80th percentile; "
            "exclude tickets already Escalated, Resolved, or Closed."
        ),
    }


def nearest_rank_index(count: int, percentile: float) -> int:
    """Return the zero-based nearest-rank position for a nonempty sequence."""
    if count <= 0 or not 0 < percentile <= 1:
        raise ValueError("Nearest-rank percentile requires count > 0 and 0 < p <= 1.")
    import math

    return math.ceil(percentile * count) - 1


def write_distribution(
    summary: dict[str, Any],
    output_path: str | Path = OUTPUT_PATH,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    summary = summarize_ticket_distribution()
    output_path = write_distribution(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nTicket distribution written to {output_path}")


if __name__ == "__main__":
    main()
