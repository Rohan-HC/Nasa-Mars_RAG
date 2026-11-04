import hashlib
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pymupdf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from compare_nasa import compare_campaigns
from document_catalog import list_documents
from document_comparison import compare_documents
from document_indexer import index_uploaded_pdf
from health_checks import check_readiness
from logging_config import configure_logging
from rag_nasa import answer_question
from service_errors import IndexingFailedError, ServiceError

configure_logging()
logger = logging.getLogger(__name__)


class CompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


app = FastAPI(
    title="NASA Mars RAG API",
    version="0.2.0",
)


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
    return response


@app.middleware("http")
async def observe_request(request: Request, call_next):
    request_id = uuid4().hex
    request.state.request_id = request_id
    started = perf_counter()
    response = await call_next(request)
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


@app.exception_handler(ServiceError)
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


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, error: HTTPException):
    codes = {
        404: "not_found",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        503: "service_unavailable",
    }
    return _error_response(
        request,
        status_code=error.status_code,
        code=codes.get(error.status_code, "request_error"),
        message=str(error.detail),
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request,
    error: RequestValidationError,
):
    return _error_response(
        request,
        status_code=422,
        code="validation_error",
        message="Request validation failed",
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, error: Exception):
    logger.error(
        "unexpected_error",
        extra={
            "request_id": _request_id(request),
            "error_code": "internal_error",
            "error_type": type(error).__name__,
        },
    )
    return _error_response(
        request,
        status_code=500,
        code="internal_error",
        message="An unexpected server error occurred",
    )


@app.post("/compare/campaigns")
def compare_nasa_campaigns(request: CompareRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty",
        )

    return compare_campaigns(question)


class CompareDocumentsRequest(BaseModel):
    document_a: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    document_b: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    question: str = Field(min_length=1, max_length=1000)


@app.post("/compare/documents")
def compare_selected_documents(request: CompareDocumentsRequest):
    document_a = request.document_a.lower()
    document_b = request.document_b.lower()

    if document_a == document_b:
        raise HTTPException(
            status_code=422,
            detail="Select two different documents",
        )

    available = {document["sha256"] for document in list_documents()}

    if document_a not in available or document_b not in available:
        raise HTTPException(
            status_code=404,
            detail="One or both documents were not found",
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty",
        )

    return compare_documents(
        question=request.question,
        document_a=document_a,
        document_b=document_b,
    )


@app.get("/", include_in_schema=False)
def home():
    html_path = Path(__file__).resolve().parent / "web" / "index.html"
    return FileResponse(html_path)


@app.get("/documents")
def get_documents():
    documents = list_documents()

    return {
        "count": len(documents),
        "documents": documents,
    }


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/health")
@app.get("/health/live")
def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
def readiness():
    result = check_readiness()
    status = "ready" if result["ready"] else "not_ready"
    return JSONResponse(
        status_code=200 if result["ready"] else 503,
        content={"status": status, **result},
    )


@app.post("/ask")
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question must not be empty",
        )

    result = answer_question(question)
    return result


UPLOAD_DIR = Path("data/uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024


@app.post("/documents/upload", status_code=201)
def upload_document(file: UploadFile = File(...)):
    hasher = hashlib.sha256()
    original_name = file.filename or ""

    if Path(original_name.replace("\\", "/")).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are allowed")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    document_id = uuid4().hex
    destination = UPLOAD_DIR / f"{document_id}.pdf"

    try:
        total_bytes = 0

        with destination.open("wb") as output:
            while block := file.file.read(1024 * 1024):
                hasher.update(block)
                total_bytes += len(block)

                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the 10 MB limit",
                    )

                output.write(block)

        # Verify that PyMuPDF can actually open the file.
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

    finally:
        file.file.close()

    file_hash = hasher.hexdigest()

    try:
        index_result = index_uploaded_pdf(
            pdf_path=destination,
            file_hash=file_hash,
            original_name=original_name,
        )

    except ValueError as error:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        destination.unlink(missing_ok=True)
        logger.error(
            "indexing_failed",
            extra={"error_type": type(error).__name__},
        )
        raise IndexingFailedError() from error

    if index_result["status"] == "already_indexed":
        destination.unlink(missing_ok=True)

    return {
        "document_id": document_id,
        "filename": original_name,
        "sha256": file_hash,
        "pages": page_count,
        "size_bytes": total_bytes,
        "index_status": index_result["status"],
        "chunks": index_result["chunks"],
    }
