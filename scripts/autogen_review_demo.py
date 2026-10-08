"""Run the two offline AutoGen review examples and save their transcript."""

from __future__ import annotations

import anyio
from pathlib import Path

from review.autogen_review import ReviewEvidence, run_review_async

P1_ANSWER = (
    "For P1, the initial response is due within 15 minutes and the target "
    "resolution time is within 4 hours."
)
EVIDENCE = ReviewEvidence(
    route="policy",
    canonical_answer=P1_ANSWER,
    grounded=True,
    fallback_triggered=False,
    evidence_ids=["KB-002"],
    facts={"severity": "P1"},
)
BAD_DRAFT = (
    "P1 tickets are resolved within 30 minutes and customers automatically "
    "receive compensation."
)


def _scenario_text(
    title: str,
    draft: str,
    reviewer_approved: bool,
    final_answer: str,
    reason: str,
    reviewer_count: int,
    turn_count: int,
    revised: bool,
) -> str:
    editor_decision = (
        "return the draft unchanged"
        if not revised
        else "replace unsupported claims with the exact evidence-supported answer"
    )
    return "\n".join(
        (
            title,
            f"Evidence IDs: {', '.join(EVIDENCE.evidence_ids)}",
            f"Draft: {draft}",
            f"Reviewer decision: {'approve' if reviewer_approved else 'reject'}",
            f"Editor decision: {editor_decision}",
            f"approved: {str(reviewer_approved).lower()}",
            f"Reviewer count: {reviewer_count}",
            f"Turn count: {turn_count}",
            f"revised: {str(revised).lower()}",
            f"Final answer: {final_answer}",
            f"Decision: {reason}",
        )
    )


async def _render_transcript() -> str:
    approved = await run_review_async(P1_ANSWER, EVIDENCE)
    corrected = await run_review_async(BAD_DRAFT, EVIDENCE)
    blocks = [
        "Batch D AutoGen review demo",
        "Runtime: AutoGen AgentChat RoundRobinGroupChat with ReplayChatCompletionClient.",
        "External model/API calls: none.",
        "Operational tools available to either reviewer role: none.",
        "",
        _scenario_text(
            "SCENARIO 1 — APPROVED DRAFT, RETURNED UNCHANGED",
            P1_ANSWER,
            approved.assessment.approved,
            approved.result.final_answer,
            approved.result.reason,
            approved.result.reviewer_count,
            approved.result.turn_count,
            approved.result.revised,
        ),
        "",
        _scenario_text(
            "SCENARIO 2 — UNSUPPORTED DRAFT, MINIMUM EVIDENCE-BASED CORRECTION",
            BAD_DRAFT,
            corrected.assessment.approved,
            corrected.result.final_answer,
            corrected.result.reason,
            corrected.result.reviewer_count,
            corrected.result.turn_count,
            corrected.result.revised,
        ),
        "",
        "Knowledge base mutation: none.",
        "Prompt/system instructions printed: none.",
    ]
    return "\n".join(blocks) + "\n"


def main() -> None:
    transcript = anyio.run(_render_transcript)
    destination = Path(__file__).resolve().parents[1] / "transcripts" / "autogen_review_demo.txt"
    destination.write_text(transcript, encoding="utf-8")
    print(transcript, end="")
    print(f"Saved: {destination.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
