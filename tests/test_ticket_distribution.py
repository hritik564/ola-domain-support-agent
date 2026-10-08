import json

from dataset import SUPPORT_TICKETS
from evaluation.ticket_distribution import (
    nearest_rank_index,
    summarize_ticket_distribution,
    write_distribution,
)


def test_dataset_distribution_statistics_and_score_threshold():
    summary = summarize_ticket_distribution()
    assert summary["ticket_count"] == len(SUPPORT_TICKETS) == 50
    assert summary["escalated_flag_counts"] == {"true": 10, "false": 40}
    assert summary["days_since_created"] == {
        "min": 0,
        "max": 30,
        "median": 15.0,
        "p80": 25.0,
        "percentile_method": "nearest_rank",
        "p80_rank_1_based": 40,
    }
    assert summary["escalation_score_distribution"] == {
        "min": 0.0,
        "max": 1.0,
        "median": 0.22166666666666665,
        "p80": 0.35,
        "p80_rank_1_based": 40,
        "recommended_count": 7,
        "eligible_ticket_count": 17,
        "not_recommended_eligible_count": 10,
        "score_at_or_above_threshold_count": 18,
        "ineligible_status_count": 33,
    }
    assert summary["recommendation_threshold"] == 0.35 * (25 / 30)
    assert summary["recommendation_threshold_derivation"] == {
        "age_p80_days": 25.0,
        "normalized_age_signal": 25.0 / 30,
        "age_weight": 0.35,
        "expression": "0.35 * (25 / 30)",
    }
    assert summary["weights"]["sum"] == 1.0


def test_nearest_rank_index_is_deterministic_for_even_sample():
    assert nearest_rank_index(50, 0.80) == 39


def test_distribution_artifact_writer(tmp_path):
    summary = summarize_ticket_distribution()
    output_path = write_distribution(summary, tmp_path / "distribution.json")
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == summary
