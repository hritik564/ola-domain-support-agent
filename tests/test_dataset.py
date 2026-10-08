from dataset import (
    CATEGORIES,
    MAX_RESOLUTION_HOURS,
    MIN_RESOLUTION_HOURS,
    STATUSES,
    SUPPORT_TICKETS,
    generate_support_tickets,
    validate_dataset,
)


def test_dataset_is_deterministic():
    assert generate_support_tickets() == generate_support_tickets()


def test_dataset_passes_capstone_constraints():
    report = validate_dataset(SUPPORT_TICKETS)
    assert report["total_records"] >= 40
    assert 10 <= report["escalated_percentage"] <= 30


def test_every_required_category_has_at_least_three_records():
    report = validate_dataset(SUPPORT_TICKETS)
    for category in CATEGORIES:
        assert report["category_counts"][category] >= 3


def test_every_required_status_is_present():
    report = validate_dataset(SUPPORT_TICKETS)
    for status in STATUSES:
        assert report["status_counts"][status] >= 1


def test_field_ranges_and_types():
    for ticket in SUPPORT_TICKETS:
        assert MIN_RESOLUTION_HOURS <= ticket["resolution_time_hours"] <= MAX_RESOLUTION_HOURS
        assert 0 <= ticket["days_since_created"] <= 30
        assert isinstance(ticket["escalated"], bool)


def test_record_ids_are_unique():
    ids = [ticket["record_id"] for ticket in SUPPORT_TICKETS]
    assert len(ids) == len(set(ids))
