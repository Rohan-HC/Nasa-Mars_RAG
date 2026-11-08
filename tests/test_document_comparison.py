import logging
from unittest.mock import Mock, patch

from langchain_core.documents import Document

import document_comparison

HASH_A = "a" * 64
HASH_B = "b" * 64


def test_comparison_replaces_uncited_claims_with_insufficient_evidence(caplog):
    model = Mock()
    model.invoke.return_value = Mock(content="The documents report different rocks.")

    def search(*, question, k, sha256=None, version_ids=None):
        filename = "a.pdf" if sha256 == HASH_A else "b.pdf"
        return [
            (
                Document(
                    page_content=f"Evidence from {filename}",
                    metadata={"filename": filename, "page": 1},
                ),
                0.1,
            )
        ]

    with caplog.at_level(logging.INFO, logger="document_comparison"):
        with (
            patch(
                "document_comparison.list_documents",
                return_value=[
                    {"sha256": HASH_A, "filename": "a.pdf"},
                    {"sha256": HASH_B, "filename": "b.pdf"},
                ],
            ),
            patch("document_comparison.search_nasa", side_effect=search),
            patch.object(document_comparison, "llm", model),
        ):
            result = document_comparison.compare_documents(
                "Compare the rocks", HASH_A, HASH_B
            )

    assert result["answer"] == "Insufficient evidence for a reliable comparison."
    assert result["validation"]["citation_valid"] is False
    assert set(result["timings_ms"]) == {"retrieval", "model", "total"}
    record = next(
        record for record in caplog.records if record.msg == "comparison_complete"
    )
    assert record.model_ms == result["timings_ms"]["model"]
