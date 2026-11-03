import re
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrieval_embeddings import NomicRetrievalEmbeddings


DB_PATH = "data/chroma"
COLLECTION = "nasa_mars_uploads"
INDEX_VERSION = "v2"


def index_uploaded_pdf(
    pdf_path: Path,
    file_hash: str,
    original_name: str,
) -> dict:
    documents = []
    heading_pattern = re.compile(
        r"(?im)^[ \t]*(references|bibliography)[ \t]*:?[ \t]*$"
    )
    in_references = False

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=False)

            if not text.strip():
                continue

            if in_references:
                sections = [("references", text)]
            else:
                match = heading_pattern.search(text)

                if match:
                    sections = [
                        ("content", text[: match.start()]),
                        ("references", text[match.end() :]),
                    ]
                    in_references = True
                else:
                    sections = [("content", text)]

            for section, section_text in sections:
                if not section_text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=section_text,
                        metadata={
                            "page": page_number,
                            "filename": original_name,
                            "sha256": file_hash,
                            "source": str(pdf_path),
                            "section": section,
                        },
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No extractable text found")

    ids = [
        f"{file_hash}:{INDEX_VERSION}:{index}" for index in range(1, len(chunks) + 1)
    ]
    expected_ids = set(ids)

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=NomicRetrievalEmbeddings(),
        persist_directory=DB_PATH,
    )

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

    if existing_ids:
        store.delete(ids=list(existing_ids))

    try:
        store.add_documents(documents=chunks, ids=ids)

        saved_ids = set(store.get(ids=ids, include=[])["ids"])
        if saved_ids != expected_ids:
            raise RuntimeError("Indexed chunk IDs do not match expected IDs")
    except Exception:
        try:
            store.delete(ids=ids)
        except Exception:
            pass
        raise

    return {
        "status": "indexed",
        "chunks": len(chunks),
    }
