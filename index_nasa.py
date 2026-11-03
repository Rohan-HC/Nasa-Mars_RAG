from pathlib import Path

import pymupdf

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from retrieval_embeddings import NomicRetrievalEmbeddings


PDF_PATH = Path("data/raw/mars_report.pdf")
DB_PATH = "data/chroma"


def load_nasa_pdf(path: Path) -> list[Document]:
    documents = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=False)

            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page": page_number,
                        "title": "Exploring the Jezero Crater Floor",
                    },
                )
            )

    return documents


if __name__ == "__main__":
    documents = load_nasa_pdf(PDF_PATH)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

    chunks = splitter.split_documents(documents)

    embeddings = NomicRetrievalEmbeddings()

    vector_store = Chroma(
        collection_name="nasa_mars_experiment",
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    ids = [f"mars_report_chunk_{index}" for index in range(1, len(chunks) + 1)]

    vector_store.add_documents(documents=chunks, ids=ids)

    print(f"Indexed {len(chunks)} chunks.")
