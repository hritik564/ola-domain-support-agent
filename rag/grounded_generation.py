"""Deterministic extractive grounded answers for MOCK_LLM-compatible runs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from rag.calibration import CALIBRATION_STRATEGIES, run_calibration
from rag.chunking import split_policy_sentences
from rag.retrieval import expand_parent_context, retrieve

FALLBACK_ANSWER = "I don't know based on the available support policies."
DEMONSTRATION_QUERIES = (
    "What is the response SLA for a P1 critical ticket?",
    "When should a repeated unresolved complaint be escalated?",
    "When can service credit be considered?",
    "What support is available outside standard business hours?",
    "How should customers be informed during a widespread outage?",
    "How do I bake a chocolate cake?",
)
_STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "do", "for", "how", "i", "in",
    "is", "it", "me", "of", "on", "or", "should", "the", "to", "what",
    "when", "who", "with",
}
_WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE_TERMINATOR = re.compile(r"[.!?][\"')\]]*$")
_TIME_VALUE = re.compile(r"\b\d+(?:\.\d+)?\s+(?:minutes?|hours?|days?)\b", re.I)


def _normalized_terms(text: str) -> set[str]:
    terms = set()
    for token in _WORD_PATTERN.findall(text.lower()):
        if token in _STOP_WORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith(("ing", "ed")):
            token = token[:-3] if token.endswith("ing") else token[:-2]
        elif len(token) > 6 and token.endswith("ion"):
            token = token[:-3]
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        if token == "inform":
            terms.add("communicat")
        elif token == "communicat":
            terms.add("inform")
        terms.add(token)
    return terms


def _complete_sentences(result: dict[str, Any]) -> list[str]:
    """Return only complete, capitalized sentences ending in punctuation."""
    metadata = result.get("metadata") or {}
    strategy = metadata.get("chunk_strategy")
    chunk_index = metadata.get("chunk_index", 0)
    sentences = split_policy_sentences(result.get("text", ""))
    complete: list[str] = []

    for index, sentence in enumerate(sentences):
        if not _SENTENCE_TERMINATOR.search(sentence):
            continue
        if not sentence[0].isupper():
            # A fixed-size chunk can begin in the middle of a sentence.
            # Lowercase-leading pieces are fragments, not policy sentences.
            continue
        if strategy == "fixed" and chunk_index > 0 and index == 0:
            # Without the preceding chunk we cannot prove that the first
            # punctuation-terminated span starts at a sentence boundary.
            continue
        complete.append(sentence)

    return complete


def _relevance_score(query_terms: set[str], sentence: str) -> int:
    sentence_terms = _normalized_terms(sentence)
    if {"response", "sla"} & query_terms and "response" not in sentence_terms:
        return 0

    priorities = {"p1", "p2", "p3", "p4"}
    requested_priorities = query_terms & priorities
    sentence_priorities = sentence_terms & priorities
    if requested_priorities and sentence_priorities and not (
        requested_priorities & sentence_priorities
    ):
        return 0

    term_weights = {
        term: 3 if term in {"response", "sla"}
        else 2 if term in priorities | {"escalat"}
        else 1
        for term in query_terms
    }
    score = sum(
        term_weights[term] for term in query_terms & sentence_terms
    )

    if requested_priorities & sentence_priorities:
        score += 2

    # A severity-specific response SLA sentence is a better match than a
    # generic SLA definition or an escalation instruction.
    if (
        "response" in query_terms
        and "response" in sentence_terms
        and _TIME_VALUE.search(sentence)
    ):
        score += 5

    return score


def _compose_from_context(
    query: str,
    results: list[dict[str, Any]],
    max_sentences: int = 3,
) -> tuple[str, set[str]]:
    """Select complete, relevant policy sentences verbatim from retrieved chunks."""
    query_terms = _normalized_terms(query)
    candidates: list[tuple[int, int, int, str, str]] = []
    for result_position, result in enumerate(results):
        chunk_id = result["chunk_id"]
        for sentence_position, sentence in enumerate(_complete_sentences(result)):
            score = _relevance_score(query_terms, sentence)
            if score:
                candidates.append(
                    (score, result_position, sentence_position, sentence, chunk_id)
                )

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[tuple[str, str]] = []
    seen_sentences: set[str] = set()
    best_score = candidates[0][0] if candidates else 0
    for score, _, _, sentence, chunk_id in candidates:
        if score < best_score:
            break
        normalized_sentence = sentence.casefold()
        if normalized_sentence in seen_sentences:
            continue
        selected.append((sentence, chunk_id))
        seen_sentences.add(normalized_sentence)
        if len(selected) >= max_sentences:
            break

    answer = " ".join(sentence for sentence, _ in selected)
    return answer, {chunk_id for _, chunk_id in selected}


def _cosine_similarity(result: dict[str, Any]) -> float:
    try:
        distance = float(result["distance"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Each retrieved result must include a numeric cosine distance.") from error
    if not math.isfinite(distance):
        raise ValueError("Retrieved cosine distance must be finite.")
    return 1.0 - distance


def generate_grounded_answer(
    query: str,
    strategy: str,
    threshold: float,
    top_k: int = 3,
    expand_parent_siblings: bool = False,
) -> dict[str, Any]:
    """Answer from retrieved policy evidence or return the exact fallback."""
    if strategy not in CALIBRATION_STRATEGIES:
        raise ValueError("strategy must be either 'fixed' or 'sentence'.")
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)):
        raise ValueError("threshold must be a finite number from empirical calibration.")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be a finite number from empirical calibration.")
    if not isinstance(expand_parent_siblings, bool):
        raise ValueError("expand_parent_siblings must be a boolean.")

    results = retrieve(query, strategy=strategy, top_k=top_k)
    top1_similarity = _cosine_similarity(results[0]) if results else None
    base: dict[str, Any] = {
        "query": query,
        "answer": FALLBACK_ANSWER,
        "grounded": False,
        "fallback_triggered": True,
        "strategy": strategy,
        "threshold": float(threshold),
        "top1_similarity": top1_similarity,
        "top1_document_id": results[0].get("document_id") if results else None,
        "top1_chunk_id": results[0].get("chunk_id") if results else None,
        "sources": [],
    }
    if top1_similarity is None or top1_similarity < float(threshold):
        return base

    context_results = (
        expand_parent_context(results, strategy=strategy)
        if expand_parent_siblings
        else [
            {**result, "retrieval_origin": "vector_search"}
            for result in results
        ]
    )
    answer, used_chunk_ids = _compose_from_context(query, context_results)
    if not answer:
        return base

    sources = []
    for result in context_results:
        if result["chunk_id"] not in used_chunk_ids:
            continue
        source: dict[str, Any] = {
            "document_id": result["document_id"],
            "chunk_id": result["chunk_id"],
            "retrieval_origin": result["retrieval_origin"],
        }
        if result["retrieval_origin"] == "vector_search":
            source["similarity"] = _cosine_similarity(result)
        sources.append(source)
    return {
        **base,
        "answer": answer,
        "grounded": True,
        "fallback_triggered": False,
        "sources": sources,
    }


def main() -> None:
    """Calibrate each strategy, then run the six required demonstrations."""
    calibration = run_calibration()
    for strategy in CALIBRATION_STRATEGIES:
        threshold_info = calibration["strategies"][strategy]["threshold"]
        print(f"\n{strategy} strategy — calibration: {threshold_info['status']}")
        if threshold_info["selected_threshold"] is None:
            print(threshold_info["message"])
            print("Demonstrations skipped because no empirical threshold was selected.")
            continue

        threshold = threshold_info["selected_threshold"]
        for index, query in enumerate(DEMONSTRATION_QUERIES, start=1):
            result = generate_grounded_answer(
                query=query,
                strategy=strategy,
                threshold=threshold,
                top_k=3,
            )
            print(f"\nDemo {index}: {json.dumps(result, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
