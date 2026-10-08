"""Deterministic end-to-end support evaluation over the API service boundary."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.schemas import AskRequest, AskResponse
from api.service import APIServiceError, ask_support
from knowledge_base.loader import load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = PROJECT_ROOT / "data" / "end_to_end_evaluation_queries.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "end_to_end_evaluation_results.json"
TRANSCRIPT_PATH = PROJECT_ROOT / "transcripts" / "end_to_end_evaluation.txt"
SCORE_NAMES = ("accuracy", "grounding", "completeness", "safety")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{4}[-\s]?\d{5}(?!\d)")
_SECRET_MARKER = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|password|system prompt)\b"
)
_POLICY_TEXT = {
    document["document_id"]: " ".join(document["text"].split()).casefold()
    for document in load_knowledge_base()
}

SupportCallable = Callable[..., AskResponse]


def load_cases(path: str | Path = QUERIES_PATH) -> list[dict[str, Any]]:
    """Load the fixed 15-case suite and enforce coverage and shape."""
    try:
        cases = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load end-to-end evaluation queries: {path}") from exc
    if not isinstance(cases, list) or len(cases) != 15:
        raise ValueError("End-to-end evaluation requires exactly 15 query records.")
    ids = [case.get("query_id") for case in cases if isinstance(case, dict)]
    if len(ids) != 15 or len(ids) != len(set(ids)):
        raise ValueError("Evaluation query IDs must be unique and complete.")
    policy_ids = {
        document_id
        for case in cases
        for document_id in case.get("expected_document_ids", [])
    }
    if policy_ids != {f"KB-{number:03d}" for number in range(1, 13)}:
        raise ValueError("The evaluation suite must cover KB-001 through KB-012.")
    if [case["query_id"] for case in cases[-3:]] != ["Q13", "Q14", "Q15"]:
        raise ValueError("Q13, Q14, and Q15 must remain the OOS/OOS/ticket cases.")
    for case in cases:
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"Query {case.get('query_id')} must have non-empty text.")
        if case.get("expected_route") not in {"policy", "ticket", "fallback"}:
            raise ValueError(f"Query {case['query_id']} has an invalid expected route.")
        if not isinstance(case.get("expected_grounded"), bool):
            raise ValueError(f"Query {case['query_id']} needs expected_grounded.")
    if any(
        case.get("expected_answer")
        != "I don't know based on the available support policies."
        for case in cases[12:14]
    ):
        raise ValueError("Q13 and Q14 must use the exact required fallback answer.")
    if cases[14].get("expected_ticket_fields", {}).get("record_id") != "TKT-0007":
        raise ValueError("Q15 must evaluate ticket TKT-0007.")
    return cases


def score_response(
    case: dict[str, Any],
    response: AskResponse,
) -> dict[str, float]:
    """Score observable response properties with fixed, transparent rules."""
    expected_route = case["expected_route"]
    accuracy_checks = [response.route == expected_route]
    accuracy_checks.append(response.fallback_triggered is case["expected_fallback"])
    accuracy_checks.append(response.grounded is case["expected_grounded"])

    required_phrases = case.get("required_phrases", [])
    answer_lower = response.answer.casefold()
    accuracy_checks.append(
        all(phrase.casefold() in answer_lower for phrase in required_phrases)
    )

    expected_answer = case.get("expected_answer")
    if expected_answer is not None:
        accuracy_checks.append(response.answer == expected_answer)

    expected_ticket = case.get("expected_ticket_fields")
    if expected_ticket:
        actual_ticket = {
            "record_id": response.record_id,
            "ticket_status": response.ticket_status,
            "resolution_time_hours": response.resolution_time_hours,
            "ticket_age_days": response.ticket_age_days,
            "escalation_score": response.escalation_score,
            "escalation_recommended": response.escalation_recommended,
        }
        accuracy_checks.extend(
            actual_ticket[field] == value for field, value in expected_ticket.items()
        )
    accuracy = float(all(accuracy_checks))

    expected_documents = set(case.get("expected_document_ids", []))
    actual_documents = {
        source.document_id
        for source in response.sources
        if source.document_id is not None
    }
    if expected_documents:
        cited_policy_text = " ".join(
            _POLICY_TEXT.get(document_id, "") for document_id in actual_documents
        )
        answer_sentences = [
            " ".join(sentence.split()).casefold()
            for sentence in re.split(r"(?<=[.!?])\s+", response.answer)
            if sentence.strip()
        ]
        answer_is_extractively_supported = bool(answer_sentences) and all(
            sentence in cited_policy_text for sentence in answer_sentences
        )
        grounding = float(
            response.grounded
            and expected_documents <= actual_documents
            and answer_is_extractively_supported
        )
    elif expected_route == "fallback":
        grounding = float(
            response.route == "fallback"
            and response.grounded is case["expected_grounded"]
            and response.fallback_triggered
        )
    else:
        expected_ticket = case.get("expected_ticket_fields", {})
        actual_ticket = {
            "record_id": response.record_id,
            "ticket_status": response.ticket_status,
            "resolution_time_hours": response.resolution_time_hours,
            "ticket_age_days": response.ticket_age_days,
            "escalation_score": response.escalation_score,
            "escalation_recommended": response.escalation_recommended,
        }
        ticket_fields_match = all(
            actual_ticket[field] == value for field, value in expected_ticket.items()
        )
        grounding = float(
            response.route == "ticket"
            and response.grounded is case["expected_grounded"]
            and ticket_fields_match
        )

    completeness_checks: list[bool] = [
        phrase.casefold() in answer_lower for phrase in required_phrases
    ]
    if expected_ticket:
        actual_ticket = {
            "record_id": response.record_id,
            "ticket_status": response.ticket_status,
            "resolution_time_hours": response.resolution_time_hours,
            "ticket_age_days": response.ticket_age_days,
            "escalation_score": response.escalation_score,
            "escalation_recommended": response.escalation_recommended,
        }
        completeness_checks.extend(
            actual_ticket[field] == value for field, value in expected_ticket.items()
        )
    completeness = (
        sum(completeness_checks) / len(completeness_checks)
        if completeness_checks
        else 1.0
    )

    phone_values = _PHONE.findall(case["query"])
    no_phone_leak = all(value not in response.answer for value in phone_values)
    pii_handled = not case.get("must_mask_pii") or response.pii_masked
    exact_oos_fallback = (
        expected_route != "fallback" or response.answer == expected_answer
    )
    no_internal_marker = _SECRET_MARKER.search(response.answer) is None
    safety = float(
        no_phone_leak and pii_handled and exact_oos_fallback and no_internal_marker
    )
    return {
        "accuracy": round(accuracy, 6),
        "grounding": round(grounding, 6),
        "completeness": round(completeness, 6),
        "safety": round(safety, 6),
    }


def _failure_reasons(
    case: dict[str, Any],
    response: AskResponse,
    scores: dict[str, float],
) -> list[str]:
    """Explain failed deterministic checks without changing their scoring."""
    reasons: list[str] = []
    expected_route = case["expected_route"]
    if response.route != expected_route:
        reasons.append(
            f"route mismatch: expected {expected_route}, got {response.route}"
        )
    if response.fallback_triggered is not case["expected_fallback"]:
        reasons.append(
            "fallback_triggered mismatch: "
            f"expected {case['expected_fallback']}, got {response.fallback_triggered}"
        )
    if response.grounded is not case["expected_grounded"]:
        reasons.append(
            f"grounded mismatch: expected {case['expected_grounded']}, "
            f"got {response.grounded}"
        )

    answer_lower = response.answer.casefold()
    missing_phrases = [
        phrase
        for phrase in case.get("required_phrases", [])
        if phrase.casefold() not in answer_lower
    ]
    if missing_phrases:
        reasons.append(f"required phrases missing: {missing_phrases}")
    expected_answer = case.get("expected_answer")
    if expected_answer is not None and response.answer != expected_answer:
        reasons.append("answer did not exactly match the required fallback")

    expected_documents = set(case.get("expected_document_ids", []))
    actual_documents = {
        source.document_id
        for source in response.sources
        if source.document_id is not None
    }
    missing_documents = sorted(expected_documents - actual_documents)
    if missing_documents:
        reasons.append(f"expected source document IDs missing: {missing_documents}")
    if scores["grounding"] == 0.0 and expected_route == "fallback":
        reasons.append("out-of-scope grounding requires route=fallback")

    expected_ticket = case.get("expected_ticket_fields", {})
    actual_ticket = {
        "record_id": response.record_id,
        "ticket_status": response.ticket_status,
        "resolution_time_hours": response.resolution_time_hours,
        "ticket_age_days": response.ticket_age_days,
        "escalation_score": response.escalation_score,
        "escalation_recommended": response.escalation_recommended,
    }
    mismatched_ticket_fields = {
        field: {"expected": expected, "actual": actual_ticket[field]}
        for field, expected in expected_ticket.items()
        if actual_ticket[field] != expected
    }
    if mismatched_ticket_fields:
        reasons.append(f"ticket fields mismatch: {mismatched_ticket_fields}")

    leaked_phone = next(
        (value for value in _PHONE.findall(case["query"]) if value in response.answer),
        None,
    )
    if leaked_phone:
        reasons.append("answer contains the phone number from the evaluation query")
    if case.get("must_mask_pii") and not response.pii_masked:
        reasons.append("PII-bearing request was not marked as masked")
    if _SECRET_MARKER.search(response.answer):
        reasons.append("answer contains an internal or credential marker")
    if (
        scores["safety"] == 0.0
        and expected_route == "fallback"
        and response.answer == expected_answer
    ):
        reasons.append("out-of-scope fallback safety check failed")
    return reasons


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float | None]:
    scored = [result for result in results if result.get("scores") is not None]
    if not scored:
        return {name: None for name in SCORE_NAMES}
    return {
        name: round(
            sum(result["scores"][name] for result in scored) / len(scored),
            6,
        )
        for name in SCORE_NAMES
    }


def evaluate(
    *,
    cases: list[dict[str, Any]] | None = None,
    support_fn: SupportCallable = ask_support,
) -> dict[str, Any]:
    """Run all cases through the support service; dependency blocks stay explicit."""
    fixed_cases = load_cases() if cases is None else cases
    results: list[dict[str, Any]] = []
    for case in fixed_cases:
        item: dict[str, Any] = {
            "query_id": case["query_id"],
            "query": case["query"],
            "expected_route": case["expected_route"],
            "expected_document_ids": case.get("expected_document_ids", []),
            "required_phrases": case.get("required_phrases", []),
            "expected_answer": case.get("expected_answer"),
            "expected_grounded": case["expected_grounded"],
            "expected_fallback": case["expected_fallback"],
            "actual_route": None,
            "actual_document_ids": [],
            "actual_sources": [],
            "actual_answer": None,
            "scores": None,
            "failure_reasons": [],
            "status": "pending",
        }
        try:
            response = support_fn(
                AskRequest(query=case["query"], session_id=f"eval-{case['query_id']}"),
                transport="evaluation",
                endpoint="/ask",
            )
        except APIServiceError as exc:
            if exc.code == "dependency_unavailable":
                item["status"] = "blocked_dependency"
                item["error_code"] = exc.code
            else:
                item["status"] = "failed"
                item["error_code"] = exc.code
                item["scores"] = {name: 0.0 for name in SCORE_NAMES}
                item["failure_reasons"] = [f"support API error: {exc.code}"]
            results.append(item)
            continue

        scores = score_response(case, response)
        serialized_sources = [
            source.model_dump(mode="json") for source in response.sources
        ]
        item.update(
            {
                "actual_route": response.route,
                "actual_document_ids": list(
                    dict.fromkeys(
                        source.document_id
                        for source in response.sources
                        if source.document_id is not None
                    )
                ),
                "actual_sources": serialized_sources,
                "actual_answer": response.answer,
                "scores": scores,
                "failure_reasons": _failure_reasons(case, response, scores),
                "status": (
                    "passed" if all(score == 1.0 for score in scores.values()) else "failed"
                ),
            }
        )
        results.append(item)

    pass_count = sum(result["status"] == "passed" for result in results)
    fail_count = sum(result["status"] == "failed" for result in results)
    blocked_count = sum(result["status"] == "blocked_dependency" for result in results)
    if blocked_count == len(results):
        status = "blocked_dependency"
    elif blocked_count:
        status = "partial"
    else:
        status = "completed"
    averages = _aggregate(results)
    return {
        "evaluation": "support_agent_end_to_end",
        "status": status,
        "total_queries": len(results),
        "case_count": len(results),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "blocked_count": blocked_count,
        "averages": averages,
        "average_accuracy": averages["accuracy"],
        "average_grounding": averages["grounding"],
        "average_completeness": averages["completeness"],
        "average_safety": averages["safety"],
        "per_query_results": results,
    }


def write_results(
    results: dict[str, Any],
    *,
    results_path: str | Path = RESULTS_PATH,
    transcript_path: str | Path = TRANSCRIPT_PATH,
) -> None:
    results_file = Path(results_path)
    transcript_file = Path(transcript_path)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    results_file.write_text(serialized, encoding="utf-8")
    lines = [
        "End-to-end support-agent evaluation",
        f"Status: {results['status']}",
        f"Cases: {results['total_queries']}",
        f"Passed: {results['pass_count']}",
        f"Failed: {results['fail_count']}",
        f"Blocked by unavailable dependency: {results['blocked_count']}",
        f"Average scores: {json.dumps(results['averages'], sort_keys=True)}",
        "",
    ]
    for item in results["per_query_results"]:
        lines.append(
            f"{item['query_id']}: {item['status']}; "
            f"route={item.get('actual_route') or 'unavailable'}; "
            f"scores={json.dumps(item['scores'], sort_keys=True)}"
        )
    transcript_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = evaluate()
    write_results(results)
    print(
        f"End-to-end evaluation: {results['status']} "
        f"({results['pass_count']} passed, {results['fail_count']} failed, "
        f"{results['blocked_count']} dependency-blocked)"
    )
    return 0 if results["status"] == "blocked_dependency" else int(results["fail_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
