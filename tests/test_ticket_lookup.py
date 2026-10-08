import pytest

from dataset import SUPPORT_TICKETS
from tools.ticket_lookup import (
    AGE_SIGNAL_WEIGHT,
    ESCALATED_FLAG_WEIGHT,
    RECOMMENDATION_ELIGIBLE_STATUSES,
    RECOMMENDATION_PERCENTILE,
    calculate_escalation_score,
    check_support_ticket_status,
    get_escalation_recommendation_threshold,
)


def test_known_record_returns_current_dataset_fields():
    ticket = SUPPORT_TICKETS[0]
    result = check_support_ticket_status(ticket["record_id"])
    assert result["record_id"] == ticket["record_id"]
    assert result["status"] == ticket["status"]
    assert result["resolution_time_hours"] == ticket["resolution_time_hours"]
    assert result["escalated"] == ticket["escalated"]


def test_lookup_score_is_deterministic_and_in_range():
    record_id = SUPPORT_TICKETS[0]["record_id"]
    first = check_support_ticket_status(record_id)
    second = check_support_ticket_status(record_id)
    assert first == second
    assert 0.0 <= first["escalation_score"] <= 1.0


def test_escalated_flag_increases_score_for_same_age():
    days = 12
    assert calculate_escalation_score(True, days) > (
        calculate_escalation_score(False, days)
    )


def test_older_ticket_increases_age_signal_and_score():
    young = calculate_escalation_score(False, 5)
    old = calculate_escalation_score(False, 25)
    assert old > young
    assert 25 / 30 > 5 / 30


def test_zero_day_and_thirty_day_age_signals_are_endpoints():
    assert calculate_escalation_score(False, 0) == 0.0
    assert calculate_escalation_score(False, 30) == 0.35


def test_recommendation_threshold_comes_from_dataset_age_p80():
    threshold = get_escalation_recommendation_threshold()
    assert threshold == 0.35 * (25 / 30)
    assert check_support_ticket_status(SUPPORT_TICKETS[0]["record_id"])[
        "recommendation_threshold"
    ] == threshold
    assert not (
        calculate_escalation_score(False, 0) >= threshold
    )
    assert calculate_escalation_score(False, 25) >= threshold
    assert calculate_escalation_score(False, 30) >= threshold
    assert calculate_escalation_score(True, 0) >= threshold


def test_recommendations_are_limited_to_unresolved_unescalated_statuses():
    assert RECOMMENDATION_ELIGIBLE_STATUSES == ("Open", "In Progress")
    inactive_ticket = next(
        ticket
        for ticket in SUPPORT_TICKETS
        if ticket["status"] in {"Escalated", "Resolved", "Closed"}
        and calculate_escalation_score(
            ticket["escalated"], ticket["days_since_created"]
        ) >= get_escalation_recommendation_threshold()
    )
    result = check_support_ticket_status(inactive_ticket["record_id"])
    assert result["escalation_recommended"] is False


def test_score_weights_sum_to_one_and_flag_weight_dominates_full_age_swing():
    assert ESCALATED_FLAG_WEIGHT + AGE_SIGNAL_WEIGHT == 1.0
    assert ESCALATED_FLAG_WEIGHT > AGE_SIGNAL_WEIGHT
    assert calculate_escalation_score(True, 0) > calculate_escalation_score(
        False, 30
    )


def test_unknown_record_id_raises_clear_deterministic_error():
    with pytest.raises(
        ValueError,
        match="Support ticket not found: TKT-9999",
    ):
        check_support_ticket_status("TKT-9999")


def test_invalid_record_id_and_age_are_rejected():
    with pytest.raises(ValueError, match="non-empty string"):
        check_support_ticket_status("")
    with pytest.raises(ValueError, match="integer from 0 to 30"):
        calculate_escalation_score(False, 31)


def test_dataset_is_the_only_ticket_source():
    record_ids = {ticket["record_id"] for ticket in SUPPORT_TICKETS}
    assert len(SUPPORT_TICKETS) == 50
    assert len(record_ids) == 50
    assert {
        check_support_ticket_status(record_id)["record_id"]
        for record_id in record_ids
    } == record_ids


def test_age_threshold_uses_nearest_rank_p80():
    assert RECOMMENDATION_PERCENTILE == 0.80
