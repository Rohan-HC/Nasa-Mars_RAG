import hashlib
from document_indexer import index_uploaded_pdf
import logging

logger = logging.getLogger(__name__)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_nasa import answer_question


app = FastAPI(
    title="NASA Mars RAG API",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


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


from pathlib import Path
from uuid import uuid4

import pymupdf
from fastapi import UploadFile, File, HTTPException


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
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Failed to index uploaded PDF %s",
            document_id,
        )
        raise HTTPException(
            status_code=503,
            detail="PDF saved, but indexing failed",
        ) from error

    return {
        "document_id": document_id,
        "filename": original_name,
        "sha256": file_hash,
        "pages": page_count,
        "size_bytes": total_bytes,
        "index_status": index_result["status"],
        "chunks": index_result["chunks"],
    }

    return {
        "document_id": document_id,
        "filename": original_name,
        "sha256": hasher.hexdigest(),
        "pages": page_count,
        "size_bytes": total_bytes,
        "status": "uploaded",
    }
