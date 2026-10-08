import pytest
from pydantic import ValidationError

from agents.schemas import SourceReference, SupportResponse


def test_policy_response_validates_as_pydantic():
    response = SupportResponse.model_validate(
        {
            "answer": "P1 tickets receive a response within one hour.",
            "route": "policy",
            "grounded": True,
            "fallback_triggered": False,
            "sources": [
                {
                    "document_id": "KB-002",
                    "chunk_id": "KB-002-S01",
                    "similarity": 0.81,
                }
            ],
        }
    )
    assert response.route == "policy"
    assert response.sources[0] == SourceReference(
        document_id="KB-002",
        chunk_id="KB-002-S01",
        similarity=0.81,
    )
    assert "retrieval_origin" not in response.model_dump()["sources"][0]


def test_parent_expansion_source_origin_is_preserved_without_similarity():
    response = SupportResponse.model_validate(
        {
            "answer": "The policy answer.",
            "route": "policy",
            "grounded": True,
            "fallback_triggered": False,
            "sources": [
                {
                    "document_id": "KB-002",
                    "chunk_id": "KB-002-sentence-000",
                    "retrieval_origin": "parent_expansion",
                }
            ],
        }
    )

    assert response.sources[0].document_id == "KB-002"
    assert response.sources[0].chunk_id == "KB-002-sentence-000"
    assert response.sources[0].retrieval_origin == "parent_expansion"
    assert response.sources[0].similarity is None
    assert (
        response.model_dump()["sources"][0]["retrieval_origin"]
        == "parent_expansion"
    )


def test_ticket_response_requires_lookup_fields():
    with pytest.raises(ValidationError, match="ticket responses require fields"):
        SupportResponse.model_validate(
            {
                "answer": "Ticket status.",
                "route": "ticket",
                "grounded": True,
                "fallback_triggered": False,
            }
        )


def test_ticket_response_requires_actual_resolution_time():
    with pytest.raises(ValidationError, match="resolution_time_hours"):
        SupportResponse.model_validate(
            {
                "answer": "Ticket TKT-0007 is In Progress.",
                "route": "ticket",
                "grounded": True,
                "fallback_triggered": False,
                "record_id": "TKT-0007",
                "ticket_status": "In Progress",
                "ticket_age_days": 13,
                "escalation_score": 0.8,
                "escalation_recommended": True,
            }
        )


def test_invalid_source_similarity_and_unknown_fields_fail_validation():
    with pytest.raises(ValidationError):
        SourceReference(document_id="KB-002", similarity=1.1)
    with pytest.raises(ValidationError):
        SupportResponse.model_validate(
            {
                "answer": "answer",
                "route": "policy",
                "grounded": True,
                "fallback_triggered": False,
                "unexpected": "unsupported",
            }
        )
