from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from api import create_app
from nasa_rag.catalog import CatalogRepository
from nasa_rag.db import Database
from nasa_rag.indexing import ExtractedChunk, IndexingService
from nasa_rag.jobs import JobQueue
from nasa_rag.security import TokenService, hash_password
from nasa_rag.worker import IngestionWorker


class RecordingExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, path):
        self.calls += 1
        return [ExtractedChunk(page=1, section="content", text=path.name)]


class MemoryDenseIndex:
    def __init__(self):
        self.chunks = {}

    def upsert(self, chunks):
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
        self.chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self.chunks.items()
            if chunk.version_id != version_id
        }


@pytest.fixture
def job_system(tmp_path):
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'jobs.db'}")
    database.create_schema()
    with database.session() as session:
        catalog = CatalogRepository(session)
        alice = catalog.create_user("alice-jobs", hash_password("alice-password"))
        bob = catalog.create_user("bob-jobs", hash_password("bob-password"))
    extractor = RecordingExtractor()
    dense = MemoryDenseIndex()
    indexing = IndexingService(
        database, extractor, dense, storage_root=tmp_path / "uploads"
    )
    queue = JobQueue(database)
    tokens = TokenService("j" * 32)
    app = create_app(
        database=database,
        token_service=tokens,
        indexing_service=indexing,
        job_queue=queue,
        upload_dir=tmp_path / "uploads",
    )
    client = TestClient(app)
    headers = {
        user.id: {"Authorization": f"Bearer {tokens.issue(user.id)}"}
        for user in (alice, bob)
    }
    return {
        "database": database,
        "alice": alice,
        "bob": bob,
        "extractor": extractor,
        "dense": dense,
        "indexing": indexing,
        "queue": queue,
        "client": client,
        "headers": headers,
        "tmp_path": tmp_path,
    }


def pdf_bytes(text="Mars job fixture"):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    content = pdf.tobytes()
    pdf.close()
    return content


def test_upload_returns_202_without_running_extraction_or_embedding(job_system):
    response = job_system["client"].post(
        "/documents/upload",
        headers=job_system["headers"][job_system["alice"].id],
        files={"file": ("queued.pdf", pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert job_system["extractor"].calls == 0
    job = job_system["queue"].get_for_owner(
        response.json()["job_id"], job_system["alice"].id
    )
    assert job.status == "queued"


def test_job_status_is_private_and_does_not_enumerate_ids(job_system):
    upload = job_system["client"].post(
        "/documents/upload",
        headers=job_system["headers"][job_system["alice"].id],
        files={"file": ("private.pdf", pdf_bytes(), "application/pdf")},
    ).json()

    denied = job_system["client"].get(
        f"/jobs/{upload['job_id']}",
        headers=job_system["headers"][job_system["bob"].id],
    )
    allowed = job_system["client"].get(
        f"/jobs/{upload['job_id']}",
        headers=job_system["headers"][job_system["alice"].id],
    )

    assert denied.status_code == 404
    assert "private.pdf" not in denied.text
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "queued"
    assert "upload_path" not in allowed.json()


def test_two_workers_cannot_claim_same_live_job(job_system):
    upload = job_system["client"].post(
        "/documents/upload",
        headers=job_system["headers"][job_system["alice"].id],
        files={"file": ("once.pdf", pdf_bytes(), "application/pdf")},
    ).json()
    now = datetime.now(UTC)

    first = job_system["queue"].claim("worker-1", lease_seconds=30, now=now)
    second = job_system["queue"].claim("worker-2", lease_seconds=30, now=now)

    assert first.id == upload["job_id"]
    assert second is None


def test_expired_lease_is_recovered_and_stale_worker_cannot_complete(job_system):
    upload = job_system["client"].post(
        "/documents/upload",
        headers=job_system["headers"][job_system["alice"].id],
        files={"file": ("recover.pdf", pdf_bytes(), "application/pdf")},
    ).json()
    now = datetime.now(UTC)
    first = job_system["queue"].claim("worker-1", lease_seconds=10, now=now)

    recovered = job_system["queue"].claim(
        "worker-2", lease_seconds=10, now=now + timedelta(seconds=11)
    )

    assert recovered.id == first.id
    assert recovered.attempt_count == 2
    assert job_system["queue"].succeed(first.id, "worker-1", chunks=1) is False


def test_heartbeat_extends_live_lease(job_system):
    job_system["client"].post(
        "/documents/upload",
        headers=job_system["headers"][job_system["alice"].id],
        files={"file": ("heartbeat.pdf", pdf_bytes(), "application/pdf")},
    )
    now = datetime.now(UTC)
    claimed = job_system["queue"].claim(
        "worker-1", lease_seconds=10, now=now
    )

    assert job_system["queue"].heartbeat(
        claimed.id, "worker-1", lease_seconds=30
    )
    assert (
        job_system["queue"].claim(
            "worker-2", lease_seconds=10, now=now + timedelta(seconds=11)
        )
        is None
    )


def test_retry_after_post_promotion_crash_is_idempotent(job_system):
    upload = job_system["client"].post(
        "/documents/upload",
        headers=job_system["headers"][job_system["alice"].id],
        files={"file": ("retry.pdf", pdf_bytes(), "application/pdf")},
    ).json()

    class CrashAfterPromotion:
        def __init__(self, delegate):
            self.delegate = delegate
            self.first = True

        def process(self, version_id):
            result = self.delegate.process(version_id)
            if self.first:
                self.first = False
                raise ConnectionError("worker lost database connection")
            return result

    worker = IngestionWorker(
        job_system["queue"],
        CrashAfterPromotion(job_system["indexing"]),
        worker_id="retry-worker",
        retry_base_seconds=0,
    )

    assert worker.run_once() == "retryable_failed"
    assert worker.run_once() == "succeeded"
    job = job_system["queue"].get_for_owner(
        upload["job_id"], job_system["alice"].id
    )
    assert job.status == "succeeded"
    assert job.attempt_count == 2
    assert len(job_system["dense"].version_chunk_ids(upload["version_id"])) == 1


def test_queued_replacement_keeps_previous_version_active(job_system):
    original_path = job_system["tmp_path"] / "original.pdf"
    original_path.write_bytes(pdf_bytes("original"))
    initial = job_system["indexing"].prepare_new_document(
        owner_id=job_system["alice"].id,
        name="logical.pdf",
        storage_path=original_path,
        content_sha256="a" * 64,
    )
    job_system["indexing"].process(initial.version_id)
    calls_before = job_system["extractor"].calls

    response = job_system["client"].put(
        f"/documents/{initial.document_id}/content",
        headers=job_system["headers"][job_system["alice"].id],
        files={"file": ("replacement.pdf", pdf_bytes("replacement"), "application/pdf")},
    )

    assert response.status_code == 202
    assert job_system["extractor"].calls == calls_before
    with job_system["database"].session() as session:
        document = CatalogRepository(session).require_document(
            job_system["alice"].id, initial.document_id
        )
        assert document.active_version_id == initial.version_id


def test_terminal_invalid_document_is_not_retried_and_source_is_removed(
    job_system,
):
    upload = job_system["client"].post(
        "/documents/upload",
        headers=job_system["headers"][job_system["alice"].id],
        files={"file": ("empty.pdf", pdf_bytes(), "application/pdf")},
    ).json()

    class InvalidDocumentService:
        def process(self, _version_id):
            raise ValueError("No extractable text found")

    worker = IngestionWorker(
        job_system["queue"],
        InvalidDocumentService(),
        worker_id="invalid-worker",
    )

    assert worker.run_once() == "failed"
    job = job_system["queue"].get_for_owner(
        upload["job_id"], job_system["alice"].id
    )
    assert job.status == "failed"
    assert job.error_class == "invalid_document"
    assert not Path(job.upload_path).exists()


def test_deletion_cancels_queued_job_and_removes_managed_upload(job_system):
    upload = job_system["client"].post(
        "/documents/upload",
        headers=job_system["headers"][job_system["alice"].id],
        files={"file": ("cancel.pdf", pdf_bytes(), "application/pdf")},
    ).json()
    job = job_system["queue"].get_for_owner(
        upload["job_id"], job_system["alice"].id
    )
    source = Path(job.upload_path)
    assert source.exists()

    response = job_system["client"].delete(
        f"/documents/{upload['document_id']}",
        headers=job_system["headers"][job_system["alice"].id],
    )

    assert response.status_code == 204
    cancelled = job_system["queue"].get_for_owner(
        upload["job_id"], job_system["alice"].id
    )
    assert cancelled.status == "cancelled"
    assert not source.exists()
