"""CrewAI tool adapter for the existing deterministic ticket lookup."""

from __future__ import annotations

import json
from typing import Type

from pydantic import BaseModel, Field

from agents.audit import begin_tool_invocation, finish_tool_invocation
from tools.crewai_compat import CrewAIBaseTool
from tools.ticket_lookup import check_support_ticket_status


class TicketLookupInput(BaseModel):
    record_id: str = Field(
        min_length=5,
        pattern=r"^TKT-\d+$",
        description="Support-ticket record ID, such as TKT-0007.",
    )


class TicketStatusTool(CrewAIBaseTool):
    name: str = "ticket_status_lookup"
    description: str = (
        "Look up the current dataset-backed status and escalation recommendation "
        "for one support ticket."
    )
    args_schema: Type[BaseModel] = TicketLookupInput

    def _run(self, record_id: str) -> str:
        invocation_id = begin_tool_invocation(
            "ticket_status_lookup",
            {"record_id": record_id},
        )
        try:
            result = check_support_ticket_status(record_id)
        except Exception as exc:
            finish_tool_invocation(
                invocation_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finish_tool_invocation(invocation_id, result=result)
        return json.dumps(result, ensure_ascii=False)
