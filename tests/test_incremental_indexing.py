from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from langchain_core.embeddings import Embeddings

from nasa_rag.catalog import CatalogRepository, DocumentAccessError
from nasa_rag.db import Database
from nasa_rag.indexing import (
    ConcurrentReplacementError,
    ExtractedChunk,
    IndexingService,
    PdfChunkExtractor,
)
from nasa_rag.retrieval import ChromaDenseIndex
from nasa_rag.security import hash_password


class RecordingExtractor:
    def __init__(self):
        self.calls: list[Path] = []

    def extract(self, path: Path) -> list[ExtractedChunk]:
        self.calls.append(path)
        return [
            ExtractedChunk(
                page=1,
                section="content",
                text=f"evidence from {path.name}",
            )
        ]


class MemoryDenseIndex:
    def __init__(self):
        self.chunks = {}
        self.upserted_versions: list[str] = []
        self.deleted_versions: list[str] = []
        self.fail_next_upsert = False
        self.fail_next_delete = False

    def upsert(self, chunks):
        if self.fail_next_upsert:
            self.fail_next_upsert = False
            raise RuntimeError("embedding unavailable")
        if chunks:
            self.upserted_versions.append(chunks[0].version_id)
        self.chunks.update({chunk.chunk_id: chunk for chunk in chunks})

    def verify(self, chunks):
        return all(self.chunks.get(chunk.chunk_id) == chunk for chunk in chunks)

    def version_chunk_ids(self, version_id):
        return {
            chunk_id
            for chunk_id, chunk in self.chunks.items()
            if chunk.version_id == version_id
        }

    def delete_version(self, version_id):
        if self.fail_next_delete:
            self.fail_next_delete = False
            raise RuntimeError("vector cleanup unavailable")
        self.deleted_versions.append(version_id)
        self.chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self.chunks.items()
            if chunk.version_id != version_id
        }


@pytest.fixture
def indexing(tmp_path):
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'catalog.db'}")
    database.create_schema()
    with database.session() as session:
        owner = CatalogRepository(session).create_user(
            "owner", hash_password("owner-password")
        )
    extractor = RecordingExtractor()
    dense = MemoryDenseIndex()
    service = IndexingService(database, extractor, dense)
    return database, owner, extractor, dense, service


def _create(service, owner_id, path, digest):
    plan = service.prepare_new_document(
        owner_id=owner_id,
        name=path.name,
        storage_path=path,
        content_sha256=digest,
    )
    result = service.process(plan.version_id)
    return plan, result


def test_unchanged_reupload_skips_extraction_and_embedding(indexing, tmp_path):
    database, owner, extractor, dense, service = indexing
    path = tmp_path / "b.pdf"
    path.write_bytes(b"b-v1")
    initial, _ = _create(service, owner.id, path, "b" * 64)
    extraction_count = len(extractor.calls)
    embedding_count = len(dense.upserted_versions)

    duplicate = service.prepare_replacement(
        owner_id=owner.id,
        document_id=initial.document_id,
        storage_path=path,
        content_sha256="b" * 64,
    )

    assert duplicate.action == "skip"
    assert duplicate.version_id == initial.version_id
    assert len(extractor.calls) == extraction_count
    assert len(dense.upserted_versions) == embedding_count


def test_non_owner_cannot_prepare_replacement(indexing, tmp_path):
    database, owner, _, _, service = indexing
    path = tmp_path / "private.pdf"
    path.write_bytes(b"private")
    plan, _ = _create(service, owner.id, path, "a" * 64)
    with database.session() as session:
        stranger = CatalogRepository(session).create_user(
            "stranger", hash_password("stranger-password")
        )

    with pytest.raises(DocumentAccessError):
        service.prepare_replacement(
            owner_id=stranger.id,
            document_id=plan.document_id,
            storage_path=path,
            content_sha256="b" * 64,
        )


def test_replacing_b_does_not_reprocess_a_or_c(indexing, tmp_path):
    _, owner, extractor, dense, service = indexing
    documents = {}
    for name, digest in (("a.pdf", "a" * 64), ("b.pdf", "b" * 64), ("c.pdf", "c" * 64)):
        path = tmp_path / name
        path.write_bytes(name.encode())
        documents[name], _ = _create(service, owner.id, path, digest)
    extractor.calls.clear()
    dense.upserted_versions.clear()
    changed_path = tmp_path / "b-v2.pdf"
    changed_path.write_bytes(b"b-v2")

    replacement = service.prepare_replacement(
        owner_id=owner.id,
        document_id=documents["b.pdf"].document_id,
        storage_path=changed_path,
        content_sha256="d" * 64,
    )
    service.process(replacement.version_id)

    assert extractor.calls == [changed_path]
    assert dense.upserted_versions == [replacement.version_id]


def test_failed_replacement_keeps_previous_version_active(indexing, tmp_path):
    database, owner, _, dense, service = indexing
    path = tmp_path / "b.pdf"
    path.write_bytes(b"b-v1")
    initial, _ = _create(service, owner.id, path, "b" * 64)
    replacement_path = tmp_path / "b-v2.pdf"
    replacement_path.write_bytes(b"b-v2")
    replacement = service.prepare_replacement(
        owner_id=owner.id,
        document_id=initial.document_id,
        storage_path=replacement_path,
        content_sha256="d" * 64,
    )
    dense.fail_next_upsert = True

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        service.process(replacement.version_id)

    with database.session() as session:
        catalog = CatalogRepository(session)
        document = catalog.require_document(owner.id, initial.document_id)
        failed = catalog.get_version(replacement.version_id)
        assert document.active_version_id == initial.version_id
        assert failed.status == "failed"
        assert catalog.allowed_active_version_ids(owner.id) == [initial.version_id]


def test_concurrent_replacement_uses_optimistic_active_version_check(
    indexing, tmp_path
):
    database, owner, _, dense, service = indexing
    original_path = tmp_path / "original.pdf"
    original_path.write_bytes(b"original")
    original, _ = _create(service, owner.id, original_path, "a" * 64)
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    first = service.prepare_replacement(
        owner_id=owner.id,
        document_id=original.document_id,
        storage_path=first_path,
        content_sha256="b" * 64,
    )
    second = service.prepare_replacement(
        owner_id=owner.id,
        document_id=original.document_id,
        storage_path=second_path,
        content_sha256="c" * 64,
    )

    service.process(first.version_id)
    with pytest.raises(ConcurrentReplacementError):
        service.process(second.version_id)

    with database.session() as session:
        document = CatalogRepository(session).require_document(
            owner.id, original.document_id
        )
        assert document.active_version_id == first.version_id
    assert dense.version_chunk_ids(second.version_id) == set()


def test_delete_hides_document_before_best_effort_derived_cleanup(
    indexing, tmp_path
):
    database, owner, _, dense, service = indexing
    path = tmp_path / "delete.pdf"
    path.write_bytes(b"delete")
    plan, _ = _create(service, owner.id, path, "e" * 64)
    dense.fail_next_delete = True

    cleanup_complete = service.delete_document(owner.id, plan.document_id)

    assert cleanup_complete is False
    with database.session() as session:
        catalog = CatalogRepository(session)
        assert catalog.list_accessible_documents(owner.id) == []
        assert catalog.allowed_active_version_ids(owner.id) == []
    assert dense.version_chunk_ids(plan.version_id)

    assert service.reconcile_derived_data() >= 1
    assert dense.version_chunk_ids(plan.version_id) == set()
    assert service.reconcile_derived_data() == 0


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5]


def test_real_pdf_sqlite_and_chroma_skip_intact_duplicate(tmp_path):
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'integration.db'}")
    database.create_schema()
    with database.session() as session:
        owner = CatalogRepository(session).create_user(
            "integration-owner", hash_password("integration-password")
        )
    pdf_path = tmp_path / "synthetic.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "SHERLOC observed a synthetic Mars target.")
    pdf.save(pdf_path)
    pdf.close()
    embeddings = CountingEmbeddings()
    dense = ChromaDenseIndex(tmp_path / "chroma", embeddings)
    service = IndexingService(database, PdfChunkExtractor(), dense)

    initial = service.prepare_new_document(
        owner_id=owner.id,
        name="synthetic.pdf",
        storage_path=pdf_path,
        content_sha256="f" * 64,
    )
    service.process(initial.version_id)
    calls_after_initial = embeddings.document_calls
    duplicate = service.prepare_replacement(
        owner_id=owner.id,
        document_id=initial.document_id,
        storage_path=pdf_path,
        content_sha256="f" * 64,
    )

    assert duplicate.action == "skip"
    assert embeddings.document_calls == calls_after_initial
    with database.session() as session:
        catalog = CatalogRepository(session)
        assert catalog.chunk_count(initial.version_id) == 1
        assert catalog.chunk_ids(initial.version_id) == dense.version_chunk_ids(
            initial.version_id
        )
