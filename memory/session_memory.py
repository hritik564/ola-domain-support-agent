"""LangChain-backed ticket reference memory isolated by session ID."""

from __future__ import annotations

import re
from threading import RLock
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

_TICKET_ID = re.compile(r"\bTKT-\d+\b", re.IGNORECASE)


class TicketSessionMemory:
    """Keep ticket references in LangChain message history per session."""

    def __init__(self) -> None:
        self._histories: dict[str, InMemoryChatMessageHistory] = {}
        self._lock = RLock()
        self._chain = RunnableWithMessageHistory(
            RunnableLambda(self._resolve),
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="output",
        )

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if not session_id:
            raise ValueError("session_id must be a non-empty string")
        with self._lock:
            return self._histories.setdefault(
                session_id,
                InMemoryChatMessageHistory(),
            )

    @staticmethod
    def _resolve(values: dict[str, Any]) -> dict[str, Any]:
        current_text = str(values.get("input", ""))
        current_ids = _TICKET_ID.findall(current_text)
        history = values.get("history", [])
        history_ids: list[str] = []
        for message in history:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                history_ids.extend(_TICKET_ID.findall(content))
        record_id = (current_ids or history_ids or [None])[-1]
        return {
            "record_id": record_id.upper() if record_id else None,
            "output": (
                f"Resolved ticket reference: {record_id.upper()}"
                if record_id
                else "No ticket reference is available in this session."
            ),
        }

    def resolve_record_id(self, session_id: str, message: str) -> str | None:
        result = self._chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}},
        )
        record_id = result.get("record_id")
        return record_id if isinstance(record_id, str) else None

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._histories.pop(session_id, None)
