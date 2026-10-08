"""Render the Batch D tool-access and risk-classification demonstration."""

from __future__ import annotations

from pathlib import Path

from governance.policy import (
    AGENT_TOOL_ACCESS,
    MEDIUM_RISK_RATIONALE,
    PROHIBITED_ACTIONS,
    RISK_CLASSIFICATION,
    validate_tool_access_policy,
)


def render_demo() -> str:
    validate_tool_access_policy()
    lines = [
        "Batch D governance demo",
        f"Risk classification: {RISK_CLASSIFICATION}",
        "Rationale:",
    ]
    lines.extend(f"- {item}" for item in MEDIUM_RISK_RATIONALE)
    lines.extend(("", "Agent tool-access matrix:"))
    for role, tools in AGENT_TOOL_ACCESS.items():
        lines.append(f"- {role}: {', '.join(tools) if tools else 'no tools'}")
    lines.extend(("", "Prohibited capabilities:"))
    lines.extend(f"- {action}" for action in PROHIBITED_ACTIONS)
    lines.extend(
        (
            "",
            "Least-autonomy check: passed.",
            "Reviewer/editor operational tools: none.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    output = render_demo()
    destination = Path(__file__).resolve().parents[1] / "transcripts" / "governance_demo.txt"
    destination.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"Saved: {destination.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
