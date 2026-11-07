from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from nasa_rag.db import Database
from nasa_rag.models import IngestionJob, utc_now


ACTIVE_JOB_STATUSES = ("queued", "running", "retryable_failed")


class JobQueue:
    def __init__(self, database: Database):
        self.database = database

    def enqueue(
        self,
        *,
        owner_id: str,
        document_id: str,
        version_id: str,
        upload_path: str,
        kind: str,
        max_attempts: int = 3,
    ) -> IngestionJob:
        with self.database.session() as session:
            existing = session.scalar(
                select(IngestionJob).where(IngestionJob.version_id == version_id)
            )
            if existing is not None:
                if existing.status not in ACTIVE_JOB_STATUSES:
                    existing.status = "queued"
                    existing.upload_path = upload_path
                    existing.kind = kind
                    existing.attempt_count = 0
                    existing.max_attempts = max_attempts
                    existing.lease_owner = None
                    existing.lease_expires_at = None
                    existing.heartbeat_at = None
                    existing.next_attempt_at = None
                    existing.error_class = None
                    existing.result_chunks = None
                    existing.started_at = None
                    existing.finished_at = None
                    existing.updated_at = utc_now()
                    session.flush()
                return existing
            job = IngestionJob(
                owner_id=owner_id,
                document_id=document_id,
                version_id=version_id,
                upload_path=upload_path,
                kind=kind,
                max_attempts=max_attempts,
                status="queued",
            )
            session.add(job)
            session.flush()
            return job

    def get_for_owner(self, job_id: str, owner_id: str) -> IngestionJob | None:
        with self.database.session() as session:
            return session.scalar(
                select(IngestionJob).where(
                    IngestionJob.id == job_id,
                    IngestionJob.owner_id == owner_id,
                )
            )

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> IngestionJob | None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        now = now or datetime.now(UTC)
        connection = self.database.engine.connect()
        try:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            session = Session(bind=connection, expire_on_commit=False)
            try:
                session.execute(
                    update(IngestionJob)
                    .where(
                        IngestionJob.status == "running",
                        IngestionJob.lease_expires_at < now,
                        IngestionJob.attempt_count >= IngestionJob.max_attempts,
                    )
                    .values(
                        status="failed",
                        error_class="lease_expired",
                        lease_owner=None,
                        lease_expires_at=None,
                        finished_at=now,
                        updated_at=now,
                    )
                )
                eligible = or_(
                    IngestionJob.status == "queued",
                    and_(
                        IngestionJob.status == "retryable_failed",
                        or_(
                            IngestionJob.next_attempt_at.is_(None),
                            IngestionJob.next_attempt_at <= now,
                        ),
                    ),
                    and_(
                        IngestionJob.status == "running",
                        IngestionJob.lease_expires_at < now,
                    ),
                )
                job = session.scalar(
                    select(IngestionJob)
                    .where(
                        eligible,
                        IngestionJob.attempt_count < IngestionJob.max_attempts,
                    )
                    .order_by(IngestionJob.created_at, IngestionJob.id)
                    .limit(1)
                )
                if job is None:
                    session.flush()
                    connection.commit()
                    return None
                job.status = "running"
                job.attempt_count += 1
                job.lease_owner = worker_id
                job.lease_expires_at = now + timedelta(seconds=lease_seconds)
                job.heartbeat_at = now
                job.next_attempt_at = None
                job.error_class = None
                job.started_at = job.started_at or now
                job.updated_at = now
                session.flush()
                connection.commit()
                session.expunge(job)
                return job
            except Exception:
                connection.rollback()
                raise
            finally:
                session.close()
        finally:
            connection.close()

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: int,
    ) -> bool:
        now = utc_now()
        with self.database.session() as session:
            result = session.execute(
                update(IngestionJob)
                .where(
                    IngestionJob.id == job_id,
                    IngestionJob.status == "running",
                    IngestionJob.lease_owner == worker_id,
                )
                .values(
                    heartbeat_at=now,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    updated_at=now,
                )
            )
            return bool(result.rowcount)

    def succeed(
        self,
        job_id: str,
        worker_id: str,
        *,
        chunks: int,
    ) -> bool:
        now = utc_now()
        with self.database.session() as session:
            result = session.execute(
                update(IngestionJob)
                .where(
                    IngestionJob.id == job_id,
                    IngestionJob.status == "running",
                    IngestionJob.lease_owner == worker_id,
                )
                .values(
                    status="succeeded",
                    result_chunks=chunks,
                    lease_owner=None,
                    lease_expires_at=None,
                    heartbeat_at=None,
                    finished_at=now,
                    updated_at=now,
                )
            )
            return bool(result.rowcount)

    def fail(
        self,
        job_id: str,
        worker_id: str,
        *,
        retryable: bool,
        error_class: str,
        retry_delay_seconds: int = 0,
    ) -> str | None:
        now = utc_now()
        with self.database.session() as session:
            job = session.scalar(
                select(IngestionJob).where(
                    IngestionJob.id == job_id,
                    IngestionJob.status == "running",
                    IngestionJob.lease_owner == worker_id,
                )
            )
            if job is None:
                return None
            can_retry = retryable and job.attempt_count < job.max_attempts
            job.status = "retryable_failed" if can_retry else "failed"
            job.error_class = error_class
            job.next_attempt_at = (
                now + timedelta(seconds=retry_delay_seconds) if can_retry else None
            )
            job.lease_owner = None
            job.lease_expires_at = None
            job.heartbeat_at = None
            job.updated_at = now
            job.finished_at = None if can_retry else now
            session.flush()
            return job.status

    def cancel_document(self, document_id: str, owner_id: str) -> int:
        now = utc_now()
        with self.database.session() as session:
            result = session.execute(
                update(IngestionJob)
                .where(
                    IngestionJob.document_id == document_id,
                    IngestionJob.owner_id == owner_id,
                    IngestionJob.status.in_(ACTIVE_JOB_STATUSES),
                )
                .values(
                    status="cancelled",
                    lease_owner=None,
                    lease_expires_at=None,
                    heartbeat_at=None,
                    finished_at=now,
                    updated_at=now,
                )
            )
            return int(result.rowcount or 0)
