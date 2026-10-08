"""Single-collection cosine retrieval and indexing smoke-test CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.embeddings import LocalEmbeddingService
from rag.vector_store import (
    DEFAULT_PERSIST_DIRECTORY,
    FIXED_COLLECTION,
    SENTENCE_COLLECTION,
    _persistent_client,
)

COLLECTION_BY_STRATEGY = {
    "fixed": FIXED_COLLECTION,
    "sentence": SENTENCE_COLLECTION,
}


def retrieve(
    query: str,
    strategy: str = "sentence",
    top_k: int = 3,
    persist_directory: str | Path | None = None,
    embedding_service: LocalEmbeddingService | None = None,
) -> list[dict[str, Any]]:
    """Retrieve from exactly one strategy collection using cosine distance.

    Chroma returns cosine distance for these collections. Similarity is
    calculated as 1 - distance; no production threshold is applied.
    """
    if strategy not in COLLECTION_BY_STRATEGY:
        raise ValueError("strategy must be either 'fixed' or 'sentence'.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    collection = _strategy_collection(strategy, persist_directory)
    count = collection.count()
    if count == 0:
        return []

    service = embedding_service or LocalEmbeddingService()
    query_vector = service.embed_query(query)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[dict[str, Any]] = []
    for rank, (chunk_id, text, metadata, distance) in enumerate(
        zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ),
        start=1,
    ):
        numeric_distance = float(distance)
        retrieved.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "document_id": metadata["document_id"],
                "text": text,
                "metadata": metadata,
                "distance": numeric_distance,
                "similarity": 1.0 - numeric_distance,
            }
        )
    return retrieved


def _strategy_collection(
    strategy: str,
    persist_directory: str | Path | None = None,
) -> Any:
    if strategy not in COLLECTION_BY_STRATEGY:
        raise ValueError("strategy must be either 'fixed' or 'sentence'.")

    client = _persistent_client(persist_directory or DEFAULT_PERSIST_DIRECTORY)
    collection_name = COLLECTION_BY_STRATEGY[strategy]
    names = {item.name for item in client.list_collections()}
    if collection_name not in names:
        raise ValueError(
            f"Collection {collection_name!r} is missing; run python -m rag.vector_store first."
        )
    collection = client.get_collection(collection_name)
    if (collection.metadata or {}).get("hnsw:space") != "cosine":
        raise ValueError(
            f"Collection {collection_name!r} is not configured for cosine distance."
        )
    return collection


def expand_parent_context(
    raw_results: list[dict[str, Any]],
    strategy: str = "sentence",
    persist_directory: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Add same-parent chunks after raw retrieval without assigning new scores.

    Raw vector hits are preserved in rank order and marked as direct hits.
    Siblings are fetched by indexed parent metadata, never by a second vector
    search, and are appended in deterministic parent/chunk order.
    """
    if strategy not in COLLECTION_BY_STRATEGY:
        raise ValueError("strategy must be either 'fixed' or 'sentence'.")
    if not raw_results:
        return []

    parent_ids: list[str] = []
    raw_chunk_ids: set[str] = set()
    for result in raw_results:
        document_id = result.get("document_id")
        chunk_id = result.get("chunk_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Every raw retrieval hit must include a document_id.")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Every raw retrieval hit must include a chunk_id.")
        if document_id not in parent_ids:
            parent_ids.append(document_id)
        raw_chunk_ids.add(chunk_id)

    direct_hits = [
        {**result, "retrieval_origin": "vector_search"}
        for result in raw_results
    ]
    collection = _strategy_collection(strategy, persist_directory)
    sibling_records = collection.get(
        where={"document_id": {"$in": parent_ids}},
        include=["documents", "metadatas"],
    )

    parent_order = {document_id: index for index, document_id in enumerate(parent_ids)}
    siblings: list[dict[str, Any]] = []
    for chunk_id, text, metadata in zip(
        sibling_records["ids"],
        sibling_records["documents"],
        sibling_records["metadatas"],
    ):
        if not isinstance(metadata, dict) or not isinstance(text, str):
            raise ValueError("Chroma returned incomplete sibling chunk data.")
        document_id = metadata.get("document_id")
        if document_id not in parent_order or chunk_id in raw_chunk_ids:
            continue
        siblings.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": text,
                "metadata": metadata,
                "retrieval_origin": "parent_expansion",
            }
        )

    def sibling_order(result: dict[str, Any]) -> tuple[int, int, str]:
        chunk_index = result["metadata"].get("chunk_index")
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
            chunk_index = 2**63 - 1
        return (
            parent_order[result["document_id"]],
            chunk_index,
            result["chunk_id"],
        )

    siblings.sort(key=sibling_order)
    return [*direct_hits, *siblings]


def main() -> None:
    """Run the two required policy questions against each independent index."""
    queries = (
        "What is the response SLA for a P1 critical ticket?",
        "When should a repeated unresolved complaint be escalated?",
    )
    for query in queries:
        print(f"\nQuery: {query}")
        for strategy in ("fixed", "sentence"):
            results = retrieve(query, strategy=strategy, top_k=3)
            print(f"\n{strategy} collection:")
            print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
