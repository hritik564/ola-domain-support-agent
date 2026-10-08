"""CrewAI tool adapter for the existing grounded-generation pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Type

from pydantic import BaseModel, Field

from agents.audit import begin_tool_invocation, finish_tool_invocation
from rag.config import (
    PRODUCTION_STRATEGY,
    PRODUCTION_THRESHOLD,
    PRODUCTION_TOP_K,
)
from rag.grounded_generation import generate_grounded_answer
from tools.crewai_compat import CrewAIBaseTool


class RAGLookupInput(BaseModel):
    query: str = Field(min_length=3, description="The customer's policy question.")


def _as_json_mapping(value: Any, description: str) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping, Pydantic model, or dataclass")
    return dict(value)


def lookup_policy(query: str) -> dict[str, Any]:
    """Use the production-selected RAG strategy and existing grounded answerer."""
    generated = _as_json_mapping(
        generate_grounded_answer(
            query=query,
            strategy=PRODUCTION_STRATEGY,
            threshold=PRODUCTION_THRESHOLD,
            top_k=PRODUCTION_TOP_K,
            expand_parent_siblings=True,
        ),
        "grounded-generation result",
    )
    sources = []
    for source in generated["sources"]:
        source = _as_json_mapping(source, "grounded-generation source")
        serialized_source = {
            "document_id": source.get("document_id"),
            "chunk_id": source.get("chunk_id"),
        }
        if "similarity" in source:
            serialized_source["similarity"] = source["similarity"]
        if "retrieval_origin" in source:
            serialized_source["retrieval_origin"] = source["retrieval_origin"]
        sources.append(serialized_source)
    return {
        "query": generated["query"],
        "answer": generated["answer"],
        "grounded": generated["grounded"],
        "fallback_triggered": generated["fallback_triggered"],
        "strategy": generated["strategy"],
        "threshold": generated["threshold"],
        "top1_similarity": generated["top1_similarity"],
        "top1_document_id": generated["top1_document_id"],
        "top1_chunk_id": generated["top1_chunk_id"],
        "sources": sources,
    }


class RAGLookupTool(CrewAIBaseTool):
    name: str = "rag_lookup"
    description: str = (
        "Retrieve grounded support-policy evidence for a customer question."
    )
    args_schema: Type[BaseModel] = RAGLookupInput

    def _run(self, query: str) -> str:
        invocation_id = begin_tool_invocation(
            "rag_lookup",
            {"query": query},
        )
        try:
            result = lookup_policy(query)
            serialized = json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
            )
        except Exception as exc:
            finish_tool_invocation(invocation_id, error=f"{type(exc).__name__}: {exc}")
            raise
        finish_tool_invocation(invocation_id, result=result)
        return serialized
