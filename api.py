from __future__ import annotations

import hashlib
import logging
import traceback
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal
from uuid import UUID, uuid4

import pymupdf
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from compare_nasa import compare_campaigns
from document_comparison import compare_documents
from document_indexer import index_uploaded_pdf
from health_checks import check_readiness
from logging_config import configure_logging
from nasa_rag.auth import AuthenticationError, authenticate_user
from nasa_rag.catalog import CatalogRepository, DocumentAccessError
from nasa_rag.config import Settings
from nasa_rag.db import Database
from nasa_rag.models import User
from nasa_rag.indexing import (
    IndexingService,
    PdfChunkExtractor,
)
from nasa_rag.jobs import JobQueue
from nasa_rag.rate_limit import SlidingWindowRateLimiter
from nasa_rag.retrieval import ChromaDenseIndex
from nasa_rag.security import InvalidTokenError, TokenService
from rag_nasa import answer_question
from retrieval_embeddings import NomicRetrievalEmbeddings
from service_errors import ServiceError, UploadStorageError


configure_logging()
logger = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)
MAX_FILE_SIZE = 10 * 1024 * 1024


def store_validated_pdf(
    file: UploadFile,
    upload_dir: Path,
    *,
    max_file_size: int | None = None,
) -> tuple[Path, str, int, int]:
    """Persist and validate one PDF, removing partial files on every failure."""
    limit = MAX_FILE_SIZE if max_file_size is None else max_file_size
    original_name = file.filename or ""
    if Path(original_name.replace("\\", "/")).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are allowed")
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid4().hex}.pdf"
    hasher = hashlib.sha256()
    try:
        total_bytes = 0
        with destination.open("wb") as output:
            while block := file.file.read(1024 * 1024):
                total_bytes += len(block)
                if total_bytes > limit:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the 10 MB limit",
                    )
                hasher.update(block)
                output.write(block)
        with pymupdf.open(stream=destination.read_bytes(), filetype="pdf") as pdf:
            page_count = len(pdf)
            if page_count == 0:
                raise ValueError("PDF has no pages")
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except (pymupdf.FileDataError, ValueError):
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail="Uploaded file is not a readable PDF"
        )
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise UploadStorageError() from error
    finally:
        file.file.close()
    return destination, hasher.hexdigest(), page_count, total_bytes


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_ids: list[UUID] | None = Field(default=None, max_length=20)


class CompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class CompareDocumentsRequest(BaseModel):
    document_a: UUID
    document_b: UUID
    question: str = Field(min_length=1, max_length=1000)


class PermissionRequest(BaseModel):
    role: Literal["reader"]


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid4().hex)


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    request_id = _request_id(request)
    response = JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )
    response.headers["X-Request-ID"] = request_id
    if status_code == 401:
        response.headers["WWW-Authenticate"] = "Bearer"
    return response


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.database.session() as session:
        yield session


def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    session: Annotated[Session, Depends(get_session)],
) -> User:
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        user_id = request.app.state.token_service.verify(credentials.credentials)
    except InvalidTokenError as error:
        raise HTTPException(status_code=401, detail="Authentication required") from error
    user = CatalogRepository(session).get_active_user(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _default_answer_function(*, question: str, allowed_version_ids: list[str]):
    return answer_question(question, allowed_version_ids=allowed_version_ids)


def _default_comparison_function(**kwargs):
    return compare_documents(**kwargs)


def create_app(
    *,
    database: Database | None = None,
    token_service: TokenService | None = None,
    answer_function: Callable | None = None,
    comparison_function: Callable | None = None,
    upload_dir: Path | None = None,
    login_limiter: SlidingWindowRateLimiter | None = None,
    user_limiter: SlidingWindowRateLimiter | None = None,
    indexing_service: IndexingService | None = None,
    job_queue: JobQueue | None = None,
) -> FastAPI:
    settings = Settings.from_env()
    application = FastAPI(title="NASA Mars RAG API", version="0.3.0")
    application.state.database = database or Database(settings.database_url)
    application.state.token_service = token_service or TokenService(
        settings.jwt_secret,
        lifetime=timedelta(minutes=settings.access_token_minutes),
    )
    application.state.answer_function = answer_function or _default_answer_function
    application.state.comparison_function = (
        comparison_function or _default_comparison_function
    )
    application.state.upload_dir = upload_dir or settings.upload_dir
    application.state.access_token_seconds = settings.access_token_minutes * 60
    application.state.login_limiter = login_limiter or SlidingWindowRateLimiter(
        limit=10,
        window_seconds=60,
    )
    application.state.user_limiter = user_limiter or SlidingWindowRateLimiter(
        limit=60,
        window_seconds=60,
    )
    application.state.indexing_service = indexing_service
    application.state.job_queue = job_queue or JobQueue(application.state.database)
    application.state.chroma_path = settings.chroma_path

    def get_indexing_service(request: Request) -> IndexingService:
        service = request.app.state.indexing_service
        if service is None:
            service = IndexingService(
                request.app.state.database,
                PdfChunkExtractor(),
                ChromaDenseIndex(
                    request.app.state.chroma_path,
                    NomicRetrievalEmbeddings(),
                ),
                storage_root=request.app.state.upload_dir,
            )
            request.app.state.indexing_service = service
        return service

    def enforce_user_limit(request: Request, user: User) -> None:
        if not request.app.state.user_limiter.allow(user.id):
            raise HTTPException(status_code=429, detail="Too many requests")

    @application.middleware("http")
    async def observe_request(request: Request, call_next):
        request_id = uuid4().hex
        request.state.request_id = request_id
        started = perf_counter()
        request.state.started_at = started
        try:
            response = await call_next(request)
        except Exception as error:
            duration_ms = round((perf_counter() - started) * 1000, 2)
            logger.info(
                "request_complete",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                    "error_type": type(error).__name__,
                },
            )
            raise
        duration_ms = round((perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    @application.exception_handler(ServiceError)
    async def service_error_handler(request: Request, error: ServiceError):
        logger.warning(
            "service_error",
            extra={
                "request_id": _request_id(request),
                "error_code": error.code,
                "error_type": type(error).__name__,
            },
        )
        return _error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
        )

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException):
        codes = {
            401: "authentication_required",
            403: "forbidden",
            404: "not_found",
            409: "conflict",
            413: "payload_too_large",
            415: "unsupported_media_type",
            422: "validation_error",
            429: "rate_limited",
            503: "service_unavailable",
        }
        return _error_response(
            request,
            status_code=error.status_code,
            code=codes.get(error.status_code, "request_error"),
            message=str(error.detail),
        )

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request,
        _error: RequestValidationError,
    ):
        return _error_response(
            request,
            status_code=422,
            code="validation_error",
            message="Request validation failed",
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception):
        duration_ms = round(
            (perf_counter() - getattr(request.state, "started_at", perf_counter()))
            * 1000,
            2,
        )
        frames = traceback.extract_tb(error.__traceback__)
        location = None
        if frames:
            frame = frames[-1]
            location = f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
        logger.error(
            "unexpected_error",
            extra={
                "request_id": _request_id(request),
                "error_code": "internal_error",
                "error_type": type(error).__name__,
                "error_location": location,
            },
        )
        response = _error_response(
            request,
            status_code=500,
            code="internal_error",
            message="An unexpected server error occurred",
        )
        response.headers["X-Response-Time-Ms"] = str(duration_ms)
        return response

    @application.get("/", include_in_schema=False)
    def home():
        return FileResponse(Path(__file__).resolve().parent / "web" / "index.html")

    @application.get("/health")
    @application.get("/health/live")
    def liveness():
        return {"status": "ok"}

    @application.get("/health/ready")
    def readiness():
        result = check_readiness()
        status = "ready" if result["ready"] else "not_ready"
        return JSONResponse(
            status_code=200 if result["ready"] else 503,
            content={"status": status, **result},
        )

    @application.post("/auth/token")
    def login(request: Request, body: LoginRequest):
        client_host = request.client.host if request.client else "unknown"
        if not request.app.state.login_limiter.allow(client_host):
            raise HTTPException(status_code=429, detail="Too many login attempts")
        try:
            with request.app.state.database.session() as session:
                user = authenticate_user(
                    session,
                    username=body.username,
                    password=body.password,
                )
                token = request.app.state.token_service.issue(user.id)
        except AuthenticationError as error:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password",
            ) from error
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": request.app.state.access_token_seconds,
        }

    @application.get("/auth/me")
    def auth_me(current_user: Annotated[User, Depends(get_current_user)]):
        return {"user_id": current_user.id, "username": current_user.username}

    @application.get("/documents")
    def get_documents(
        current_user: Annotated[User, Depends(get_current_user)],
        session: Annotated[Session, Depends(get_session)],
    ):
        documents = CatalogRepository(session).list_accessible_documents(
            current_user.id
        )
        payload = [
            {
                "document_id": document.id,
                "name": document.name,
                "status": document.status,
                "role": "owner" if document.owner_id == current_user.id else "reader",
            }
            for document in documents
        ]
        return {"count": len(payload), "documents": payload}

    @application.get("/documents/{document_id}")
    def get_document(
        document_id: UUID,
        current_user: Annotated[User, Depends(get_current_user)],
        session: Annotated[Session, Depends(get_session)],
    ):
        catalog = CatalogRepository(session)
        try:
            document = catalog.require_document(current_user.id, str(document_id))
        except DocumentAccessError as error:
            raise HTTPException(status_code=404, detail="Document not found") from error
        version = catalog.get_version(document.active_version_id)
        return {
            "document_id": document.id,
            "name": document.name,
            "status": document.status,
            "role": "owner" if document.owner_id == current_user.id else "reader",
            "active_version": (
                {
                    "version_id": version.id,
                    "sha256": version.content_sha256,
                    "pages": version.page_count,
                    "chunks": version.chunk_count,
                }
                if version is not None
                else None
            ),
        }

    @application.get("/jobs/{job_id}")
    def get_job(
        job_id: UUID,
        request: Request,
        current_user: Annotated[User, Depends(get_current_user)],
    ):
        job = request.app.state.job_queue.get_for_owner(
            str(job_id), current_user.id
        )
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job.id,
            "document_id": job.document_id,
            "version_id": job.version_id,
            "kind": job.kind,
            "status": job.status,
            "attempt_count": job.attempt_count,
            "error_class": job.error_class,
            "chunks": job.result_chunks,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }

    @application.put("/documents/{document_id}/permissions/{username}")
    def grant_permission(
        document_id: UUID,
        username: str,
        body: PermissionRequest,
        current_user: Annotated[User, Depends(get_current_user)],
        session: Annotated[Session, Depends(get_session)],
    ):
        catalog = CatalogRepository(session)
        try:
            catalog.require_document(current_user.id, str(document_id), owner=True)
        except DocumentAccessError as error:
            raise HTTPException(status_code=404, detail="Document not found") from error
        target = catalog.get_user_by_username(username)
        if target is None:
            raise HTTPException(status_code=404, detail="Document or user not found")
        try:
            catalog.grant_reader(current_user.id, str(document_id), target.id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail="Reader cannot be granted") from error
        return {
            "document_id": str(document_id),
            "username": target.username,
            "role": body.role,
        }

    @application.delete(
        "/documents/{document_id}/permissions/{username}", status_code=204
    )
    def revoke_permission(
        document_id: UUID,
        username: str,
        current_user: Annotated[User, Depends(get_current_user)],
        session: Annotated[Session, Depends(get_session)],
    ):
        catalog = CatalogRepository(session)
        try:
            catalog.require_document(current_user.id, str(document_id), owner=True)
        except DocumentAccessError as error:
            raise HTTPException(status_code=404, detail="Document not found") from error
        target = catalog.get_user_by_username(username)
        if target is not None:
            catalog.revoke_reader(current_user.id, str(document_id), target.id)
        return Response(status_code=204)

    @application.post("/ask")
    def ask(
        request: Request,
        body: AskRequest,
        current_user: Annotated[User, Depends(get_current_user)],
        session: Annotated[Session, Depends(get_session)],
    ):
        enforce_user_limit(request, current_user)
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question must not be empty")
        catalog = CatalogRepository(session)
        requested = (
            [str(document_id) for document_id in body.document_ids]
            if body.document_ids is not None
            else None
        )
        if requested is not None:
            for document_id in requested:
                try:
                    catalog.require_document(current_user.id, document_id)
                except DocumentAccessError as error:
                    raise HTTPException(
                        status_code=404, detail="Document not found"
                    ) from error
        allowed_versions = catalog.allowed_active_version_ids(
            current_user.id,
            requested,
        )
        return request.app.state.answer_function(
            question=question,
            allowed_version_ids=allowed_versions,
        )

    @application.post("/compare/documents")
    def compare_selected_documents(
        request: Request,
        body: CompareDocumentsRequest,
        current_user: Annotated[User, Depends(get_current_user)],
        session: Annotated[Session, Depends(get_session)],
    ):
        enforce_user_limit(request, current_user)
        document_a = str(body.document_a)
        document_b = str(body.document_b)
        if document_a == document_b:
            raise HTTPException(status_code=422, detail="Select two different documents")
        catalog = CatalogRepository(session)
        try:
            first = catalog.require_document(current_user.id, document_a)
            second = catalog.require_document(current_user.id, document_b)
            first_version = catalog.active_version_for_document(
                current_user.id, document_a
            )
            second_version = catalog.active_version_for_document(
                current_user.id, document_b
            )
        except DocumentAccessError as error:
            raise HTTPException(
                status_code=404,
                detail="One or both documents were not found",
            ) from error
        return request.app.state.comparison_function(
            question=body.question.strip(),
            document_a=document_a,
            document_b=document_b,
            version_ids={
                document_a: first_version.id,
                document_b: second_version.id,
            },
            filenames={document_a: first.name, document_b: second.name},
        )

    @application.post("/compare/campaigns")
    def compare_nasa_campaigns(
        body: CompareRequest,
        current_user: Annotated[User, Depends(get_current_user)],
        session: Annotated[Session, Depends(get_session)],
    ):
        versions = CatalogRepository(session).allowed_active_version_ids(
            current_user.id
        )
        return compare_campaigns(
            body.question.strip(),
            allowed_version_ids=versions,
        )

    @application.post("/documents/upload", status_code=202)
    def upload_document(
        request: Request,
        current_user: Annotated[User, Depends(get_current_user)],
        file: UploadFile = File(...),
    ):
        enforce_user_limit(request, current_user)
        original_name = file.filename or ""
        destination, file_hash, page_count, total_bytes = store_validated_pdf(
            file,
            request.app.state.upload_dir,
        )
        try:
            service = get_indexing_service(request)
            plan = service.prepare_new_document(
                owner_id=current_user.id,
                name=original_name,
                storage_path=destination,
                content_sha256=file_hash,
            )
            job = request.app.state.job_queue.enqueue(
                owner_id=current_user.id,
                document_id=plan.document_id,
                version_id=plan.version_id,
                upload_path=str(destination),
                kind="new_document",
            )
        except ValueError as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            destination.unlink(missing_ok=True)
            logger.error("job_enqueue_failed", extra={"error_type": type(error).__name__})
            raise HTTPException(
                status_code=503, detail="Ingestion could not be queued"
            ) from error
        return {
            "job_id": job.id,
            "document_id": plan.document_id,
            "version_id": plan.version_id,
            "filename": original_name,
            "pages": page_count,
            "size_bytes": total_bytes,
            "status": job.status,
        }

    @application.put("/documents/{document_id}/content")
    def replace_document_content(
        document_id: UUID,
        request: Request,
        current_user: Annotated[User, Depends(get_current_user)],
        file: UploadFile = File(...),
    ):
        enforce_user_limit(request, current_user)
        destination, file_hash, page_count, total_bytes = store_validated_pdf(
            file,
            request.app.state.upload_dir,
        )
        try:
            service = get_indexing_service(request)
            plan = service.prepare_replacement(
                owner_id=current_user.id,
                document_id=str(document_id),
                storage_path=destination,
                content_sha256=file_hash,
                check_integrity=False,
            )
            job = request.app.state.job_queue.enqueue(
                owner_id=current_user.id,
                document_id=plan.document_id,
                version_id=plan.version_id,
                upload_path=str(destination),
                kind="replacement",
            )
        except DocumentAccessError as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=404, detail="Document not found") from error
        except ValueError as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            destination.unlink(missing_ok=True)
            logger.error("job_enqueue_failed", extra={"error_type": type(error).__name__})
            raise HTTPException(
                status_code=503, detail="Ingestion could not be queued"
            ) from error
        if Path(job.upload_path) != destination:
            destination.unlink(missing_ok=True)
        return JSONResponse(
            status_code=202,
            content={
            "job_id": job.id,
            "document_id": plan.document_id,
            "version_id": plan.version_id,
            "pages": page_count,
            "size_bytes": total_bytes,
            "status": job.status,
            },
        )

    @application.delete("/documents/{document_id}", status_code=204)
    def delete_document(
        document_id: UUID,
        request: Request,
        current_user: Annotated[User, Depends(get_current_user)],
    ):
        with request.app.state.database.session() as session:
            try:
                CatalogRepository(session).require_document(
                    current_user.id, str(document_id), owner=True
                )
            except DocumentAccessError as error:
                raise HTTPException(
                    status_code=404, detail="Document not found"
                ) from error
        request.app.state.job_queue.cancel_document(
            str(document_id), current_user.id
        )
        try:
            cleanup_complete = get_indexing_service(request).delete_document(
                current_user.id,
                str(document_id),
            )
        except DocumentAccessError as error:
            raise HTTPException(status_code=404, detail="Document not found") from error
        if not cleanup_complete:
            logger.warning(
                "derived_cleanup_pending",
                extra={"error_type": "DerivedCleanupIncomplete"},
            )
        return Response(status_code=204)

    return application


app = create_app()
