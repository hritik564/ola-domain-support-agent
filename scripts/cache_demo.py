"""Demonstrate normalized, session-scoped cache hits and safe bypasses."""

from __future__ import annotations

from pathlib import Path

from cache.normalized_cache import NormalizedResponseCache, build_cache_key


def render_demo() -> str:
    cache: NormalizedResponseCache[str] = NormalizedResponseCache(max_entries=8)
    calls = 0
    session_id = "cache-demo-session"

    def lookup(query: str) -> tuple[str, bool]:
        nonlocal calls
        key, eligibility = build_cache_key(query, session_id=session_id)
        if key is None:
            calls += 1
            return f"bypass:{eligibility}", False
        existing = cache.get(key)
        if existing.hit and existing.value is not None:
            return existing.value, True
        calls += 1
        answer = f"deterministic answer for {query.strip().casefold()}"
        cache.put(key, answer)
        return answer, False

    first_query = "What is the P1 SLA?"
    normalized_query = "  WHAT   IS THE p1 sla? "
    different_query = "What is the P2 SLA?"
    first_answer, first_hit = lookup(first_query)
    first_cache_size = cache.size
    normalized_answer, normalized_hit = lookup(normalized_query)
    normalized_cache_size = cache.size
    different_answer, different_hit = lookup(different_query)
    different_cache_size = cache.size
    pii_key, pii_reason = build_cache_key(
        "What is the P1 SLA for 9876543210?",
        session_id=session_id,
    )
    followup_key, followup_reason = build_cache_key(
        "Should that ticket be escalated?",
        session_id=session_id,
    )
    other_session_key, _ = build_cache_key(
        "What is the P1 SLA?",
        session_id="another-session",
    )
    first_key, _ = build_cache_key(
        first_query,
        session_id=session_id,
    )
    normalized_key, _ = build_cache_key(
        normalized_query,
        session_id=session_id,
    )
    return "\n".join(
        (
            "Batch D normalized cache demo",
            "Storage: process-local bounded LRU; keys are SHA-256 hashes.",
            f"First policy lookup: hit={str(first_hit).lower()}, answer={first_answer}",
            f"First cache size: {first_cache_size}",
            (
                "Whitespace/case-normalized repeat: "
                f"hit={str(normalized_hit).lower()}, answer={normalized_answer}"
            ),
            f"Normalized query uses same hashed key: {first_key == normalized_key}",
            f"Cache size after normalized hit: {normalized_cache_size}",
            f"Different query: hit={str(different_hit).lower()}, answer={different_answer}",
            f"Cache size after different query: {different_cache_size}",
            f"First key SHA-256 prefix: {first_key[:12] if first_key else 'none'}",
            f"Underlying deterministic lookup calls: {calls}",
            f"PII request bypass: {pii_key is None} ({pii_reason})",
            f"Session-reference request bypass: {followup_key is None} ({followup_reason})",
            f"Different session receives a different key: {first_key != other_session_key}",
            "Ticket IDs and session-dependent references are not cached.",
            "No raw normalized query is stored in the cache key.",
            "",
        )
    )


def main() -> None:
    output = render_demo()
    destination = Path(__file__).resolve().parents[1] / "transcripts" / "cache_demo.txt"
    destination.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"Saved: {destination.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
