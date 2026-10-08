import chromadb
import pytest

from knowledge_base.loader import load_knowledge_base
from rag.chunking import chunk_by_sentence, chunk_fixed_size
from rag.embeddings import EMBEDDING_MODEL_NAME, LocalEmbeddingService
from rag.vector_store import (
    FIXED_COLLECTION,
    PROJECT_COLLECTIONS,
    SENTENCE_COLLECTION,
    build_indexes,
)


@pytest.fixture(scope="module")
def embedding_service():
    return LocalEmbeddingService()


def test_local_embeddings_are_repeatable_and_dimensionally_consistent(
    embedding_service,
):
    text = "P1 tickets receive an initial response within 15 minutes."
    first = embedding_service.embed_documents([text])[0]
    second = embedding_service.embed_documents([text])[0]
    query = embedding_service.embed_query(text)

    assert first == pytest.approx(second, abs=1e-6)
    assert len(first) > 0
    assert len(query) == len(first)
    assert embedding_service.model_name == EMBEDDING_MODEL_NAME


def test_build_indexes_is_separate_idempotent_and_scoped(tmp_path, embedding_service):
    stats = build_indexes(
        persist_directory=tmp_path,
        embedding_service=embedding_service,
    )
    client = chromadb.PersistentClient(path=str(tmp_path))
    names = {collection.name for collection in client.list_collections()}

    assert names == set(PROJECT_COLLECTIONS)
    assert stats["documents_loaded"] == 12
    assert stats["fixed"]["collection"] == FIXED_COLLECTION
    assert stats["sentence"]["collection"] == SENTENCE_COLLECTION

    fixed_collection = client.get_collection(FIXED_COLLECTION)
    sentence_collection = client.get_collection(SENTENCE_COLLECTION)
    fixed_records = fixed_collection.get(include=["metadatas"])
    sentence_records = sentence_collection.get(include=["metadatas"])

    assert stats["fixed"]["stored_records"] == len(
        chunk_fixed_size(load_knowledge_base())
    )
    assert stats["sentence"]["stored_records"] == len(
        chunk_by_sentence(load_knowledge_base())
    )
    assert fixed_collection.count() == stats["fixed"]["chunks_created"]
    assert sentence_collection.count() == stats["sentence"]["chunks_created"]
    assert all(
        metadata["chunk_strategy"] == "fixed"
        and metadata["document_id"]
        for metadata in fixed_records["metadatas"]
    )
    assert all(
        metadata["chunk_strategy"] == "sentence"
        and metadata["document_id"]
        for metadata in sentence_records["metadatas"]
    )

    repeated = build_indexes(
        persist_directory=tmp_path,
        embedding_service=embedding_service,
    )
    assert repeated["fixed"]["stored_records"] == stats["fixed"]["stored_records"]
    assert repeated["sentence"]["stored_records"] == stats["sentence"]["stored_records"]

    unrelated = client.create_collection("unrelated_test_collection")
    assert unrelated.name == "unrelated_test_collection"
    build_indexes(
        reset=True,
        persist_directory=tmp_path,
        embedding_service=embedding_service,
    )
    names_after_reset = {collection.name for collection in client.list_collections()}
    assert names_after_reset == set(PROJECT_COLLECTIONS) | {
        "unrelated_test_collection"
    }
