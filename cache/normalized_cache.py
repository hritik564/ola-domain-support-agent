"""Small in-memory cache for stateless support answers."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar

from api.logging import mask_log_text

T = TypeVar("T")

_TICKET_ID = re.compile(r"\bTKT-\d+\b", re.IGNORECASE)
_SESSION_REFERENCE = re.compile(
    r"\b(?:it|this|that|these|those|same)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CacheLookup(Generic[T]):
    hit: bool
    value: T | None
    size: int


def normalize_query(query: str) -> str:
    """Normalize case and whitespace without stripping IDs or punctuation."""
    return " ".join(query.split()).casefold()


def build_cache_key(
    query: str,
    *,
    session_id: str | None,
) -> tuple[str | None, str]:
    """Return a non-reversible key, or a reason this query must bypass caching."""
    safe_query, query_has_pii = mask_log_text(query)
    safe_session, session_has_pii = mask_log_text(session_id)
    if query_has_pii or session_has_pii:
        return None, "pii_detected"

    normalized = normalize_query(safe_query or "")
    if not normalized:
        return None, "empty_query"
    if _TICKET_ID.search(normalized):
        return None, "ticket_id_present"
    if _SESSION_REFERENCE.search(normalized):
        return None, "session_reference_present"

    # Session scope prevents a cached result from crossing conversation boundaries.
    material = f"{safe_session or ''}\0{normalized}".encode("utf-8")
    return hashlib.sha256(material).hexdigest(), "eligible"


class NormalizedResponseCache(Generic[T]):
    """Thread-safe bounded LRU cache; only hashed keys are retained."""

    def __init__(self, max_entries: int = 128) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive.")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, T] = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> CacheLookup[T]:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
                return CacheLookup(hit=True, value=value, size=len(self._entries))
            return CacheLookup(hit=False, value=None, size=len(self._entries))

    def put(self, key: str, value: T) -> int:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return len(self._entries)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
