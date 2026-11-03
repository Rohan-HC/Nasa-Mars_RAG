from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from retrieval_embeddings import NomicRetrievalEmbeddings


DB_PATH = "data/chroma"
COLLECTION = "nasa_mars_uploads"
INDEX_VERSION = "v1"


def index_uploaded_pdf(
    pdf_path: Path,
    file_hash: str,
    original_name: str,
) -> dict:

    # 1. Convert PDF pages into LangChain documents
    documents = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=False)

            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number,
                        "filename": original_name,
                        "sha256": file_hash,
                        "source": str(pdf_path),
                    },
                )
            )

    # 2. Split pages into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No extractable text found")

    # 3. Generate deterministic identifiers
    ids = [
        f"{file_hash}:{INDEX_VERSION}:{index}" for index in range(1, len(chunks) + 1)
    ]

    expected_ids = set(ids)

    # 4. Connect to persistent Chroma
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=NomicRetrievalEmbeddings(),
        persist_directory=DB_PATH,
    )

    # 5. Check existing records for this file
    existing_ids = set(
        store.get(
            where={"sha256": file_hash},
            include=[],
        )["ids"]
    )

    if existing_ids == expected_ids:
        return {
            "status": "already_indexed",
            "chunks": len(chunks),
        }

    # Remove incomplete or outdated records
    # belonging to this exact file hash.
    if existing_ids:
        store.delete(ids=list(existing_ids))

    try:
        store.add_documents(documents=chunks, ids=ids)

        saved_ids = set(store.get(ids=ids, include=[])["ids"])
        if saved_ids != expected_ids:
            raise RuntimeError("Indexed chunk IDs do not match expected IDs")

    except Exception:
        # Best-effort cleanup of a partially written index.
        try:
            store.delete(ids=ids)
        except Exception:
            pass
        raise

    return {
        "status": "indexed",
        "chunks": len(chunks),
    }
