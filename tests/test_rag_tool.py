import json

from agents.audit import audit_context, new_audit
from agents.mock_llm import _parse_tool_result_payload
from rag.config import PRODUCTION_THRESHOLD
from tools import rag_tool


def test_rag_adapter_uses_production_sentence_settings(monkeypatch):
    calls = {}

    def generated(**kwargs):
        calls.update(kwargs)
        return {
            "query": kwargs["query"],
            "answer": "A grounded policy sentence.",
            "grounded": True,
            "fallback_triggered": False,
            "strategy": kwargs["strategy"],
            "threshold": kwargs["threshold"],
            "top1_similarity": 0.8,
            "top1_document_id": "KB-002",
            "top1_chunk_id": "KB-002-S01",
            "sources": [{"document_id": "KB-002", "chunk_id": "KB-002-S01"}],
        }

    monkeypatch.setattr(rag_tool, "generate_grounded_answer", generated)
    result = rag_tool.lookup_policy("What is the response SLA?")
    assert calls == {
        "query": "What is the response SLA?",
        "strategy": "sentence",
        "threshold": PRODUCTION_THRESHOLD,
        "top_k": 5,
        "expand_parent_siblings": True,
    }
    assert result["answer"] == "A grounded policy sentence."
    assert result["top1_document_id"] == "KB-002"
    assert result["top1_chunk_id"] == "KB-002-S01"
    assert result["sources"][0]["document_id"] == "KB-002"


def test_rag_tool_invocation_records_actual_adapter_call(monkeypatch):
    monkeypatch.setattr(
        rag_tool,
        "generate_grounded_answer",
        lambda **kwargs: {
            "query": kwargs["query"],
            "answer": "A grounded policy sentence.",
            "grounded": True,
            "fallback_triggered": False,
            "strategy": kwargs["strategy"],
            "threshold": kwargs["threshold"],
            "top1_similarity": 0.8,
            "top1_document_id": "KB-002",
            "top1_chunk_id": "KB-002-S01",
            "sources": [],
        },
    )
    audit = new_audit()
    with audit_context(audit):
        raw = rag_tool.RAGLookupTool()._run("What is the response SLA?")
    evidence = json.loads(raw)
    assert evidence["grounded"] is True
    assert audit.tool_invocation_count == 1
    assert audit.tool_invocations[0].tool_name == "rag_lookup"
    assert audit.tool_invocations[0].arguments == {
        "query": "What is the response SLA?"
    }
    assert audit.tool_invocations[0].status == "completed"


def test_actual_rag_tool_adapter_result_normalizes_as_json_evidence(monkeypatch):
    monkeypatch.setattr(
        rag_tool,
        "generate_grounded_answer",
        lambda **kwargs: {
            "query": kwargs["query"],
            "answer": "P1 response SLA is 15 minutes and resolution SLA is 4 hours.",
            "grounded": True,
            "fallback_triggered": False,
            "strategy": kwargs["strategy"],
            "threshold": kwargs["threshold"],
            "top1_similarity": 0.8,
            "top1_document_id": "KB-002",
            "top1_chunk_id": "KB-002-S01",
            "sources": [
                {
                    "document_id": "KB-002",
                    "chunk_id": "KB-002-S01",
                    "similarity": 0.8,
                }
            ],
        },
    )
    query = "What is the response SLA for a P1 ticket?"
    raw_result = rag_tool.RAGLookupTool()._run(query)
    assert isinstance(raw_result, str)
    evidence = _parse_tool_result_payload(raw_result)
    assert evidence["query"] == query
    assert evidence["grounded"] is True
    assert evidence["fallback_triggered"] is False
    assert evidence["strategy"] == "sentence"
    assert evidence["threshold"] == PRODUCTION_THRESHOLD
    assert evidence["top1_document_id"] == "KB-002"
    assert evidence["top1_chunk_id"] == "KB-002-S01"
    assert evidence["sources"][0]["document_id"] == "KB-002"


def test_parent_expanded_source_keeps_origin_without_invented_similarity(monkeypatch):
    monkeypatch.setattr(
        rag_tool,
        "generate_grounded_answer",
        lambda **kwargs: {
            "query": kwargs["query"],
            "answer": (
                "For P1, the initial response is due within 15 minutes and the "
                "target resolution time is within 4 hours."
            ),
            "grounded": True,
            "fallback_triggered": False,
            "strategy": kwargs["strategy"],
            "threshold": kwargs["threshold"],
            "top1_similarity": 0.576289,
            "top1_document_id": "KB-003",
            "top1_chunk_id": "KB-003-sentence-000",
            "sources": [
                {
                    "document_id": "KB-002",
                    "chunk_id": "KB-002-sentence-000",
                    "retrieval_origin": "parent_expansion",
                }
            ],
        },
    )

    result = json.loads(
        rag_tool.RAGLookupTool()._run(
            "What is the response SLA for a P1 critical ticket?"
        )
    )

    assert result["top1_document_id"] == "KB-003"
    assert result["top1_chunk_id"] == "KB-003-sentence-000"
    assert result["sources"] == [
        {
            "document_id": "KB-002",
            "chunk_id": "KB-002-sentence-000",
            "retrieval_origin": "parent_expansion",
        }
    ]
    assert "similarity" not in result["sources"][0]
