"""Empirical, strategy-specific retrieval threshold calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from rag.retrieval import retrieve

CALIBRATION_QUERIES = (
    {
        "query_id": "Q1",
        "query": "What is the response SLA for a P1 critical ticket?",
        "expected_scope": "in_scope",
        "expected_document_ids": ["KB-002"],
    },
    {
        "query_id": "Q2",
        "query": "When should a repeated unresolved complaint be escalated?",
        "expected_scope": "in_scope",
        "expected_document_ids": ["KB-007", "KB-003"],
    },
    {
        "query_id": "Q3",
        "query": "When can service credit be considered?",
        "expected_scope": "in_scope",
        "expected_document_ids": ["KB-008"],
    },
    {
        "query_id": "OOS1",
        "query": "Who won the FIFA World Cup?",
        "expected_scope": "out_of_scope",
        "expected_document_ids": [],
    },
    {
        "query_id": "OOS2",
        "query": "How do I bake a chocolate cake?",
        "expected_scope": "out_of_scope",
        "expected_document_ids": [],
    },
)
CALIBRATION_STRATEGIES = ("fixed", "sentence")
THRESHOLD_FORMULA = (
    "(max_out_of_scope_similarity + min_in_scope_similarity) / 2"
)
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "calibration_results.json"


def select_empirical_threshold(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the midpoint only when observed in-scope and out-of-scope scores separate."""
    in_scope = [
        float(item["top1_similarity"])
        for item in measurements
        if item.get("expected_scope") == "in_scope"
        and item.get("top1_similarity") is not None
    ]
    out_of_scope = [
        float(item["top1_similarity"])
        for item in measurements
        if item.get("expected_scope") == "out_of_scope"
        and item.get("top1_similarity") is not None
    ]
    result: dict[str, Any] = {
        "status": "insufficient_data",
        "min_in_scope_similarity": min(in_scope) if in_scope else None,
        "max_out_of_scope_similarity": max(out_of_scope) if out_of_scope else None,
        "selected_threshold": None,
        "formula": THRESHOLD_FORMULA,
    }

    complete_measurements = bool(measurements) and all(
        item.get("expected_scope") in {"in_scope", "out_of_scope"}
        and item.get("top1_similarity") is not None
        for item in measurements
    )
    if not complete_measurements or not in_scope or not out_of_scope:
        result["message"] = (
            "Calibration needs a measured top-1 similarity for every query and "
            "results from both in-scope and out-of-scope queries."
        )
        return result

    min_in_scope = result["min_in_scope_similarity"]
    max_out_of_scope = result["max_out_of_scope_similarity"]
    if max_out_of_scope < min_in_scope:
        result.update(
            status="separated",
            selected_threshold=(max_out_of_scope + min_in_scope) / 2,
            message="Measured score ranges are separated; the midpoint is selected.",
        )
    else:
        result.update(
            status="overlap_review_required",
            message=(
                "Calibration scores overlap; expand the query set or require "
                "manual review. No threshold was selected."
            ),
        )
    return result


def run_calibration(
    retrieve_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Measure top-1 results independently for every query and strategy."""
    retrieval_function = retrieve_fn or retrieve
    strategy_results: dict[str, Any] = {}

    for strategy in CALIBRATION_STRATEGIES:
        measurements: list[dict[str, Any]] = []
        for calibration_query in CALIBRATION_QUERIES:
            results = retrieval_function(
                calibration_query["query"],
                strategy=strategy,
                top_k=1,
            )
            top1 = results[0] if results else None
            distance = float(top1["distance"]) if top1 is not None else None
            similarity = 1.0 - distance if distance is not None else None
            if similarity is not None and not math.isfinite(similarity):
                raise ValueError("Retrieval returned a non-finite cosine distance.")

            measurements.append(
                {
                    "query_id": calibration_query["query_id"],
                    "query": calibration_query["query"],
                    "expected_scope": calibration_query["expected_scope"],
                    "expected_document_ids": list(
                        calibration_query["expected_document_ids"]
                    ),
                    "top1_document_id": (
                        top1.get("document_id") if top1 is not None else None
                    ),
                    "top1_chunk_id": (
                        top1.get("chunk_id") if top1 is not None else None
                    ),
                    "top1_distance": distance,
                    "top1_similarity": similarity,
                }
            )
        strategy_results[strategy] = {
            "measurements": measurements,
            "threshold": select_empirical_threshold(measurements),
        }

    return {
        "strategies": strategy_results,
        "threshold_method": THRESHOLD_FORMULA,
        "production_strategy_selected": False,
    }


def write_calibration_results(
    results: dict[str, Any],
    output_path: str | Path = DEFAULT_RESULTS_PATH,
) -> Path:
    """Write actual measured calibration results to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    results = run_calibration()
    output_path = write_calibration_results(results)
    print(f"Calibration results: {output_path}")
    for strategy in CALIBRATION_STRATEGIES:
        details = results["strategies"][strategy]
        print(f"\n{strategy} strategy:")
        for measurement in details["measurements"]:
            print(
                f"{measurement['query_id']}: {measurement['query']} | "
                f"scope={measurement['expected_scope']} | "
                f"top1={measurement['top1_document_id']} "
                f"({measurement['top1_chunk_id']}) | "
                f"distance={measurement['top1_distance']} | "
                f"similarity={measurement['top1_similarity']}"
            )
        threshold = details["threshold"]
        print(f"Minimum in-scope similarity: {threshold['min_in_scope_similarity']}")
        print(
            "Maximum out-of-scope similarity: "
            f"{threshold['max_out_of_scope_similarity']}"
        )
        print(f"Calibration status: {threshold['status']}")
        print(f"Selected threshold: {threshold['selected_threshold']}")
        if threshold["status"] != "separated":
            print(threshold["message"])
    print("\nNo overall production strategy is selected in this sprint.")


if __name__ == "__main__":
    main()
