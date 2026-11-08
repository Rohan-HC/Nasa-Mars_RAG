from __future__ import annotations

import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from logging_config import configure_logging
from nasa_rag.config import Settings
from nasa_rag.db import Database
from nasa_rag.indexing import ConcurrentReplacementError, IndexingService, PdfChunkExtractor
from nasa_rag.jobs import JobQueue
from nasa_rag.retrieval import ChromaDenseIndex
from retrieval_embeddings import NomicRetrievalEmbeddings

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(
        self,
        queue: JobQueue,
        indexing_service,
        *,
        worker_id: str,
        lease_seconds: int = 180,
        retry_base_seconds: int = 5,
    ):
        self.queue = queue
        self.indexing_service = indexing_service
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.retry_base_seconds = retry_base_seconds

    def run_once(self) -> str | None:
        job = self.queue.claim(
            self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if job is None:
            return None
        started = time.perf_counter()
        try:
            with self._heartbeat(job.id):
                result = self.indexing_service.process(job.version_id)
            if not self.queue.succeed(
                job.id,
                self.worker_id,
                chunks=result.chunks,
            ):
                return "lease_lost"
            self._remove_redundant_upload(job.upload_path, job.version_id)
            logger.info(
                "ingestion_job_complete",
                extra={
                    "job_id": job.id,
                    "worker_id": self.worker_id,
                    "stage": "complete",
                    "attempt_count": job.attempt_count,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            return "succeeded"
        except ValueError:
            status = self.queue.fail(
                job.id,
                self.worker_id,
                retryable=False,
                error_class="invalid_document",
            )
            if status == "failed":
                Path(job.upload_path).unlink(missing_ok=True)
            return status
        except ConcurrentReplacementError:
            return self._record_retry(job, "concurrent_replacement")
        except Exception as error:
            logger.warning(
                "ingestion_job_failed",
                extra={
                    "job_id": job.id,
                    "worker_id": self.worker_id,
                    "stage": "index",
                    "attempt_count": job.attempt_count,
                    "failure_category": type(error).__name__,
                    "error_type": type(error).__name__,
                },
            )
            return self._record_retry(job, "dependency_failure")

    def run_forever(self, *, poll_seconds: float = 1.0) -> None:
        while True:
            if self.run_once() is None:
                time.sleep(poll_seconds)

    def _record_retry(self, job, error_class: str) -> str | None:
        delay = self.retry_base_seconds * (2 ** max(job.attempt_count - 1, 0))
        status = self.queue.fail(
            job.id,
            self.worker_id,
            retryable=True,
            error_class=error_class,
            retry_delay_seconds=delay,
        )
        if status == "failed":
            Path(job.upload_path).unlink(missing_ok=True)
        return status

    @contextmanager
    def _heartbeat(self, job_id: str):
        stopped = threading.Event()
        interval = max(1.0, self.lease_seconds / 3)

        def renew() -> None:
            while not stopped.wait(interval):
                if not self.queue.heartbeat(
                    job_id,
                    self.worker_id,
                    lease_seconds=self.lease_seconds,
                ):
                    return

        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=interval + 1)

    def _remove_redundant_upload(self, upload_path: str, version_id: str) -> None:
        from nasa_rag.catalog import CatalogRepository

        with self.queue.database.session() as session:
            version = CatalogRepository(session).get_version(version_id)
            active_path = Path(version.storage_path) if version is not None else None
        candidate = Path(upload_path)
        if active_path is not None and candidate != active_path:
            candidate.unlink(missing_ok=True)


def build_worker() -> IngestionWorker:
    settings = Settings.from_env()
    database = Database(settings.database_url)
    queue = JobQueue(database)
    indexing = IndexingService(
        database,
        PdfChunkExtractor(),
        ChromaDenseIndex(
            settings.chroma_path,
            NomicRetrievalEmbeddings(settings.ollama_base_url),
        ),
        storage_root=settings.upload_dir,
    )
    worker_id = os.getenv(
        "NASA_RAG_WORKER_ID",
        f"{socket.gethostname()}-{uuid4().hex[:8]}",
    )
    return IngestionWorker(queue, indexing, worker_id=worker_id)


def main() -> None:
    configure_logging()
    build_worker().run_forever()


if __name__ == "__main__":
    main()
