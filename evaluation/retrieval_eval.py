"""Document-level precision and recall for both RAG chunking strategies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from rag.retrieval import retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = PROJECT_ROOT / "data" / "retrieval_evaluation_queries.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "retrieval_evaluation_results.json"
STRATEGIES = ("fixed", "sentence")
TOP_K = 3
CRITICAL_QUERY_IDS = ("Q1", "Q2")

Retriever = Callable[..., list[dict[str, Any]]]


def load_evaluation_queries(
    path: str | Path = QUERIES_PATH,
) -> list[dict[str, Any]]:
    """Read and validate the fixed five-query evaluation set."""
    try:
        queries = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load retrieval evaluation queries: {path}") from error

    if not isinstance(queries, list) or len(queries) != 5:
        raise ValueError("Retrieval evaluation requires exactly five query records.")

    query_ids: list[str] = []
    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("Each retrieval evaluation query must be an object.")
        if not all(key in query for key in ("query_id", "query", "relevant_document_ids")):
            raise ValueError(f"Incomplete retrieval evaluation query: {query!r}")
        if not isinstance(query["query_id"], str) or not query["query_id"]:
            raise ValueError("Evaluation query IDs must be non-empty strings.")
        if not isinstance(query["query"], str) or not query["query"].strip():
            raise ValueError("Evaluation query text must be non-empty.")
        relevant = query["relevant_document_ids"]
        if (
            not isinstance(relevant, list)
            or not relevant
            or any(not isinstance(item, str) or not item for item in relevant)
            or len(relevant) != len(set(relevant))
        ):
            raise ValueError(
                f"Query {query['query_id']} needs unique relevant document IDs."
            )
        query_ids.append(query["query_id"])

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Retrieval evaluation query IDs must be unique.")
    return queries


def deduplicate_preserving_order(document_ids: Iterable[str]) -> list[str]:
    """Return each parent ID once, preserving its first retrieval position."""
    seen: set[str] = set()
    unique: list[str] = []
    for document_id in document_ids:
        if document_id not in seen:
            seen.add(document_id)
            unique.append(document_id)
    return unique


def evaluate_query(
    query_record: dict[str, Any],
    strategy: str,
    retrieve_fn: Retriever,
) -> dict[str, Any]:
    """Score one query after mapping retrieved chunks to unique parent documents."""
    if strategy not in STRATEGIES:
        raise ValueError("strategy must be either 'fixed' or 'sentence'.")

    retrieved = retrieve_fn(
        query_record["query"],
        strategy=strategy,
        top_k=TOP_K,
    )
    if not isinstance(retrieved, list):
        raise ValueError("Retriever must return a list of result records.")

    chunk_ids: list[str] = []
    parent_ids: list[str] = []
    for item in retrieved:
        chunk_id = item.get("chunk_id")
        document_id = item.get("document_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Every retrieved chunk must have a non-empty chunk_id.")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(
                "Every retrieved chunk must map to a non-empty parent document_id."
            )
        chunk_ids.append(chunk_id)
        parent_ids.append(document_id)

    unique_parent_ids = deduplicate_preserving_order(parent_ids)
    if not unique_parent_ids:
        raise ValueError(
            f"Cannot calculate precision for {query_record['query_id']}: "
            "retrieval returned no parent documents."
        )

    relevant_ids = list(query_record["relevant_document_ids"])
    relevant_set = set(relevant_ids)
    true_positive_ids = [
        document_id for document_id in unique_parent_ids
        if document_id in relevant_set
    ]
    true_positives = len(true_positive_ids)
    precision_denominator = len(unique_parent_ids)
    recall_denominator = len(relevant_ids)

    return {
        "query_id": query_record["query_id"],
        "query": query_record["query"],
        "strategy": strategy,
        "relevant_document_ids": relevant_ids,
        "raw_retrieved_chunk_ids": chunk_ids,
        "retrieved_parent_document_ids_before_dedup": parent_ids,
        "unique_retrieved_parent_document_ids": unique_parent_ids,
        "true_positive_document_ids": true_positive_ids,
        "true_positives": true_positives,
        "precision_numerator": true_positives,
        "precision_denominator": precision_denominator,
        "precision": true_positives / precision_denominator,
        "recall_numerator": true_positives,
        "recall_denominator": recall_denominator,
        "recall": true_positives / recall_denominator,
    }


def _aggregate(per_query: list[dict[str, Any]]) -> dict[str, float]:
    if not per_query:
        raise ValueError("Cannot aggregate an empty evaluation result.")
    count = len(per_query)
    return {
        "mean_precision": sum(item["precision"] for item in per_query) / count,
        "mean_recall": sum(item["recall"] for item in per_query) / count,
    }


def select_recommended_strategy(
    strategies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the documented recall, precision, then secondary-evidence order."""
    if any(
        strategy not in strategies
        or not strategies[strategy].get("per_query")
        or strategies[strategy].get("aggregate") is None
        for strategy in STRATEGIES
    ):
        return {
            "status": "pending_external_validation",
            "strategy": None,
            "threshold": None,
            "reason": "A production strategy cannot be selected before both strategies have measured results.",
        }

    fixed = strategies["fixed"]
    sentence = strategies["sentence"]
    fixed_aggregate = fixed["aggregate"]
    sentence_aggregate = sentence["aggregate"]

    for metric in ("mean_recall", "mean_precision"):
        fixed_value = fixed_aggregate[metric]
        sentence_value = sentence_aggregate[metric]
        if not math.isclose(fixed_value, sentence_value, abs_tol=1e-12):
            winner = "fixed" if fixed_value > sentence_value else "sentence"
            return _recommendation_for(winner, strategies, metric)

    fixed_rank_one = _critical_rank_one_hits(fixed["per_query"])
    sentence_rank_one = _critical_rank_one_hits(sentence["per_query"])
    if fixed_rank_one != sentence_rank_one:
        winner = "fixed" if fixed_rank_one > sentence_rank_one else "sentence"
        recommendation = _recommendation_for(
            winner,
            strategies,
            "critical_rank_one_hits",
        )
        recommendation["reason"] = (
            "Mean recall and precision tied; "
            f"{winner} has more rank-one hits on critical queries "
            f"({max(fixed_rank_one, sentence_rank_one)} vs "
            f"{min(fixed_rank_one, sentence_rank_one)})."
        )
        return recommendation

    fixed_irrelevant = _irrelevant_document_count(fixed["per_query"])
    sentence_irrelevant = _irrelevant_document_count(sentence["per_query"])
    if fixed_irrelevant != sentence_irrelevant:
        winner = "fixed" if fixed_irrelevant < sentence_irrelevant else "sentence"
        recommendation = _recommendation_for(
            winner,
            strategies,
            "irrelevant_parent_document_count",
        )
        recommendation["reason"] = (
            "Mean recall, precision, and critical rank-one hits tied; "
            f"{winner} returned fewer irrelevant parent documents "
            f"({min(fixed_irrelevant, sentence_irrelevant)} vs "
            f"{max(fixed_irrelevant, sentence_irrelevant)})."
        )
        return recommendation

    return {
        "status": "tie_review_required",
        "strategy": None,
        "threshold": None,
        "reason": (
            "Both strategies tie on mean recall, mean precision, critical "
            "rank-one hits, and irrelevant parent-document count; do not choose "
            "a strategy arbitrarily."
        ),
    }


def _recommendation_for(
    winner: str,
    strategies: dict[str, dict[str, Any]],
    selection_basis: str,
) -> dict[str, Any]:
    threshold = _measured_calibration_threshold(winner)
    winner_aggregate = strategies[winner]["aggregate"]
    other_strategy = "sentence" if winner == "fixed" else "fixed"
    other_aggregate = strategies[other_strategy]["aggregate"]
    if threshold is None:
        return {
            "status": "pending_matching_calibration_threshold",
            "strategy": None,
            "threshold": None,
            "reason": (
                f"{winner} led on {selection_basis}, but its separately measured "
                "calibration threshold is unavailable; no production setting "
                "was selected."
            ),
        }
    winner_precision = winner_aggregate["mean_precision"]
    winner_recall = winner_aggregate["mean_recall"]
    other_precision = other_aggregate["mean_precision"]
    other_recall = other_aggregate["mean_recall"]
    if selection_basis == "mean_recall":
        reason = (
            f"{winner} is recommended: mean recall was {winner_recall:.4f} "
            f"versus {other_recall:.4f}, and mean precision was "
            f"{winner_precision:.4f} versus {other_precision:.4f}. "
            f"Use the matching calibrated threshold of {threshold!r}."
        )
    elif selection_basis == "mean_precision":
        reason = (
            f"Mean recall tied at {winner_recall:.4f}; {winner} is recommended "
            f"for higher mean precision ({winner_precision:.4f} versus "
            f"{other_precision:.4f}). Use its calibrated threshold of "
            f"{threshold!r}."
        )
    elif selection_basis == "critical_rank_one_hits":
        reason = (
            f"Mean recall and precision tied at {winner_recall:.4f} and "
            f"{winner_precision:.4f}. {winner} had more correct rank-one "
            "results on Q1/Q2; use its corresponding calibrated threshold "
            f"of {threshold!r}."
        )
    else:
        reason = (
            f"Mean recall and precision tied at {winner_recall:.4f} and "
            f"{winner_precision:.4f}. {winner} returned fewer irrelevant "
            "parent documents; use its corresponding calibrated threshold "
            f"of {threshold!r}."
        )
    return {
        "status": "recommended",
        "strategy": winner,
        "threshold": threshold,
        "metrics": {
            "mean_precision": winner_precision,
            "mean_recall": winner_recall,
        },
        "reason": reason,
    }


def _measured_calibration_threshold(strategy: str) -> float | None:
    calibration_path = PROJECT_ROOT / "data" / "calibration_results.json"
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        threshold = calibration["strategies"][strategy]["threshold"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(threshold, dict):
        return None
    if threshold.get("status") != "separated":
        return None
    selected = threshold.get("selected_threshold")
    if isinstance(selected, bool) or not isinstance(selected, (int, float)):
        return None
    if not math.isfinite(float(selected)):
        return None
    return float(selected)


def _critical_rank_one_hits(per_query: list[dict[str, Any]]) -> int:
    return sum(
        item["query_id"] in CRITICAL_QUERY_IDS
        and bool(item["unique_retrieved_parent_document_ids"])
        and item["unique_retrieved_parent_document_ids"][0]
        in item["relevant_document_ids"]
        for item in per_query
    )


def _irrelevant_document_count(per_query: list[dict[str, Any]]) -> int:
    return sum(
        sum(
            document_id not in set(item["relevant_document_ids"])
            for document_id in item["unique_retrieved_parent_document_ids"]
        )
        for item in per_query
    )


def evaluate_retrieval(
    retrieve_fn: Retriever | None = None,
    queries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the same five-query evaluation against both independent collections."""
    retrieval_function = retrieve_fn or retrieve
    query_set = queries if queries is not None else load_evaluation_queries()
    if not query_set:
        raise ValueError("At least one retrieval evaluation query is required.")

    strategy_results: dict[str, Any] = {}
    for strategy in STRATEGIES:
        per_query = [
            evaluate_query(query, strategy, retrieval_function)
            for query in query_set
        ]
        strategy_results[strategy] = {
            "status": "completed",
            "per_query": per_query,
            "aggregate": _aggregate(per_query),
            "critical_rank_one_hits": _critical_rank_one_hits(per_query),
            "irrelevant_parent_document_count": _irrelevant_document_count(
                per_query
            ),
        }

    return {
        "status": "completed",
        "top_k": TOP_K,
        "query_count": len(query_set),
        "queries": query_set,
        "strategies": strategy_results,
        "production_strategy_recommendation": select_recommended_strategy(
            strategy_results
        ),
    }


def pending_evaluation(reason: str) -> dict[str, Any]:
    """Represent unavailable Chroma measurements without inventing scores."""
    queries = load_evaluation_queries()
    return {
        "status": "pending_chromadb_validation",
        "top_k": TOP_K,
        "query_count": len(queries),
        "queries": queries,
        "strategies": {
            strategy: {
                "status": "not_run",
                "per_query": [],
                "aggregate": None,
            }
            for strategy in STRATEGIES
        },
        "production_strategy_recommendation": {
            "status": "pending_external_validation",
            "strategy": None,
            "threshold": None,
            "reason": reason,
        },
    }


def write_results(
    results: dict[str, Any],
    output_path: str | Path = RESULTS_PATH,
) -> Path:
    path = Path(output_path)
    if path.resolve() == RESULTS_PATH.resolve() and path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed_external_validation":
            raise RuntimeError(
                "Externally validated retrieval results are locked; "
                "write new measurements to a separate output path."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if existing.get("status") == "completed_external_validation":
            print(
                "Externally validated retrieval results are locked; "
                f"preserving {RESULTS_PATH}."
            )
            return
    try:
        results = evaluate_retrieval()
    except RuntimeError as error:
        if "ChromaDB is required" not in str(error):
            raise
        results = pending_evaluation(str(error))
        output_path = write_results(results)
        print(f"Retrieval evaluation pending: {error}")
        print(f"Pending result written to {output_path}")
        return

    output_path = write_results(results)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nEvaluation result written to {output_path}")


if __name__ == "__main__":
    main()
