from unittest.mock import patch

from document_indexer import INDEX_VERSION, index_uploaded_pdf


class FakeStore:
    def __init__(self, existing_ids=()):
        self.ids = set(existing_ids)
        self.added_documents = []
        self.deleted_ids = []

    def get(self, *, where=None, ids=None, include=None):
        if ids is not None:
            return {"ids": [item for item in ids if item in self.ids]}
        return {"ids": list(self.ids)}

    def add_documents(self, *, documents, ids):
        self.added_documents.extend(documents)
        self.ids.update(ids)

    def delete(self, *, ids):
        self.deleted_ids.extend(ids)
        self.ids.difference_update(ids)


def test_indexer_detects_exact_duplicate_without_embedding(pdf_file):
    file_hash = "a" * 64
    expected_id = f"{file_hash}:{INDEX_VERSION}:1"
    store = FakeStore(existing_ids=[expected_id])

    with (
        patch("document_indexer.Chroma", return_value=store),
        patch("document_indexer.NomicRetrievalEmbeddings", return_value=object()),
    ):
        result = index_uploaded_pdf(pdf_file, file_hash, "mars.pdf")

    assert result == {"status": "already_indexed", "chunks": 1}
    assert store.added_documents == []


def test_indexer_replaces_stale_chunks_and_preserves_source_metadata(pdf_file):
    file_hash = "b" * 64
    stale_id = f"{file_hash}:v1:1"
    store = FakeStore(existing_ids=[stale_id])

    with (
        patch("document_indexer.Chroma", return_value=store),
        patch("document_indexer.NomicRetrievalEmbeddings", return_value=object()),
    ):
        result = index_uploaded_pdf(pdf_file, file_hash, "mars.pdf")

    assert result == {"status": "indexed", "chunks": 1}
    assert store.deleted_ids == [stale_id]
    assert store.added_documents[0].metadata["filename"] == "mars.pdf"
    assert store.added_documents[0].metadata["section"] == "content"
    assert store.added_documents[0].metadata["sha256"] == file_hash
