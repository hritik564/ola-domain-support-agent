import json
import re
import shutil
from pathlib import Path

import pytest

from knowledge_base.loader import (
    KNOWLEDGE_BASE_DIR,
    REQUIRED_DOCUMENT_IDS,
    load_knowledge_base,
)

EXPECTED_TOPICS = {
    "ticket-priority classification rules",
    "response and resolution targets by severity",
    "severity-based escalation paths",
    "refund eligibility and compensation documentation",
    "approved customer support channels",
    "support desk hours and holiday coverage",
    "identifying and handling repeat complaints",
    "service credit eligibility and approval",
    "customer satisfaction feedback after resolution",
    "VIP support prioritisation and safeguards",
    "communication during widespread outages",
    "retention, access, and privacy for ticket records",
}


def _policy_sentences(text):
    return [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]


def test_exactly_twelve_documents_load():
    assert len(load_knowledge_base()) == 12


def test_all_required_document_ids_exist_and_are_unique():
    document_ids = [record["document_id"] for record in load_knowledge_base()]
    assert set(document_ids) == set(REQUIRED_DOCUMENT_IDS)
    assert len(document_ids) == len(set(document_ids))


def test_every_required_topic_is_represented():
    assert {record["topic"] for record in load_knowledge_base()} == EXPECTED_TOPICS


def test_every_manifest_file_exists():
    records = load_knowledge_base()
    assert all((KNOWLEDGE_BASE_DIR / record["filename"]).is_file() for record in records)


def test_every_document_has_nonempty_policy_text():
    assert all(record["text"].strip() for record in load_knowledge_base())


def test_each_document_has_two_to_five_policy_sentences():
    for record in load_knowledge_base():
        sentences = _policy_sentences(record["text"])
        assert 2 <= len(sentences) <= 5, record["filename"]


def test_knowledge_base_has_no_external_links():
    for record in load_knowledge_base():
        assert not re.search(r"https?://", record["text"], flags=re.IGNORECASE)


def test_loader_returns_stable_manifest_order():
    first_load = load_knowledge_base()
    second_load = load_knowledge_base()
    assert first_load == second_load
    assert [record["document_id"] for record in first_load] == list(
        REQUIRED_DOCUMENT_IDS
    )


def test_loader_rejects_duplicate_document_ids(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    shutil.copytree(KNOWLEDGE_BASE_DIR, tmp_path / "knowledge_base")
    entries = json.loads(
        (tmp_path / "knowledge_base" / "manifest.json").read_text(encoding="utf-8")
    )
    entries[1]["document_id"] = entries[0]["document_id"]
    manifest_path.write_text(json.dumps(entries), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate document IDs"):
        load_knowledge_base(manifest_path)
