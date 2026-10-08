"""Deterministic support-ticket dataset for the Ola Support Agent capstone.

Task 1 requirements covered:
- >= 40 records
- all required categories/statuses represented
- every category appears >= 3 times
- escalated=True rate between 10% and 30%
- fixed seed; no hand-edited records
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

SEED = 1
NUM_RECORDS = 50

CATEGORIES = [
    "Billing",
    "Technical Issue",
    "Account Access",
    "Product Defect",
    "General Inquiry",
]
CATEGORY_WEIGHTS = [0.22, 0.26, 0.18, 0.14, 0.20]

STATUSES = [
    "Open",
    "In Progress",
    "Escalated",
    "Resolved",
    "Closed",
]
STATUS_WEIGHTS = [0.18, 0.27, 0.10, 0.27, 0.18]

MIN_RESOLUTION_HOURS = 1
MAX_RESOLUTION_HOURS = 120

# Non-"Escalated" tickets still have a small probability of having been
# escalated earlier in their lifecycle. Tickets whose current status is
# "Escalated" are always marked escalated=True.
NON_ESCALATED_STATUS_ESCALATION_PROBABILITY = 0.08


def generate_support_tickets(
    seed: int = SEED,
    num_records: int = NUM_RECORDS,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tickets: list[dict[str, Any]] = []

    for index in range(1, num_records + 1):
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        days_since_created = rng.randint(0, 30)
        resolution_time_hours = rng.randint(
            MIN_RESOLUTION_HOURS, MAX_RESOLUTION_HOURS
        )

        escalated = (
            status == "Escalated"
            or rng.random() < NON_ESCALATED_STATUS_ESCALATION_PROBABILITY
        )

        tickets.append(
            {
                "record_id": f"TKT-{index:04d}",
                "category": category,
                "status": status,
                "resolution_time_hours": resolution_time_hours,
                "days_since_created": days_since_created,
                "escalated": escalated,
            }
        )

    return tickets


SUPPORT_TICKETS = generate_support_tickets()


def validate_dataset(tickets: list[dict[str, Any]]) -> dict[str, Any]:
    required_fields = {
        "record_id",
        "category",
        "status",
        "resolution_time_hours",
        "days_since_created",
        "escalated",
    }

    assert len(tickets) >= 40, "Dataset must contain at least 40 records."
    assert len({ticket["record_id"] for ticket in tickets}) == len(tickets), (
        "record_id values must be unique."
    )

    for ticket in tickets:
        assert set(ticket) == required_fields, (
            f"Unexpected fields for {ticket['record_id']}: {set(ticket)}"
        )
        assert ticket["category"] in CATEGORIES
        assert ticket["status"] in STATUSES
        assert MIN_RESOLUTION_HOURS <= ticket["resolution_time_hours"] <= MAX_RESOLUTION_HOURS
        assert 0 <= ticket["days_since_created"] <= 30
        assert isinstance(ticket["escalated"], bool)

    category_counts = Counter(ticket["category"] for ticket in tickets)
    status_counts = Counter(ticket["status"] for ticket in tickets)

    for category in CATEGORIES:
        assert category_counts[category] >= 3, (
            f"{category} must appear at least 3 times."
        )

    for status in STATUSES:
        assert status_counts[status] >= 1, (
            f"{status} must appear at least once."
        )

    escalated_count = sum(ticket["escalated"] for ticket in tickets)
    escalated_percentage = (escalated_count / len(tickets)) * 100

    assert 10 <= escalated_percentage <= 30, (
        f"Escalated percentage must be 10%-30%; got {escalated_percentage:.1f}%."
    )

    return {
        "total_records": len(tickets),
        "category_counts": dict(category_counts),
        "status_counts": dict(status_counts),
        "escalated_count": escalated_count,
        "escalated_percentage": escalated_percentage,
    }


def export_json(
    tickets: list[dict[str, Any]],
    output_path: str | Path = "data/support_tickets.json",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    return output


def print_report(report: dict[str, Any]) -> None:
    print(f"Total tickets: {report['total_records']}")
    print("\nCategory counts:")
    for category in CATEGORIES:
        print(f"  {category}: {report['category_counts'][category]}")

    print("\nStatus counts:")
    for status in STATUSES:
        print(f"  {status}: {report['status_counts'][status]}")

    print(
        f"\nEscalated tickets: {report['escalated_count']} "
        f"({report['escalated_percentage']:.1f}%)"
    )


if __name__ == "__main__":
    report = validate_dataset(SUPPORT_TICKETS)
    export_json(SUPPORT_TICKETS)
    print_report(report)
