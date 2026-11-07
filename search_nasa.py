from pathlib import Path

from langchain_chroma import Chroma
from retrieval_embeddings import NomicRetrievalEmbeddings


DB_PATH = "data/chroma"
COLLECTION = "nasa_mars_uploads"


def search_nasa(
    question: str,
    k: int = 3,
    filename: str | None = None,
    sha256: str | None = None,
    version_ids: list[str] | None = None,
):
    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty")

    if k < 1:
        raise ValueError("k must be at least 1")

    if not version_ids:
        return []

    if not Path(DB_PATH).exists():
        raise FileNotFoundError("NASA vector database not found")

    embeddings = NomicRetrievalEmbeddings()

    vector_store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    filters = [{"section": "content"}]
    filters.append({"version_id": {"$in": sorted(set(version_ids))}})

    if filename:
        filters.append({"filename": filename})

    if sha256:
        filters.append({"sha256": sha256})

    metadata_filter = (
        filters[0]
        if len(filters) == 1
        else {"$and": filters}
    )

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
        filter=metadata_filter,
    )

    return results


if __name__ == "__main__":
    question = input("Ask about NASA Mars research: ")

    results = search_nasa(question)

    for rank, (document, distance) in enumerate(results, start=1):
        print(f"\nResult {rank} (distance: {distance:.4f})")

        title = (
            document.metadata.get("title")
            or document.metadata.get("filename")
            or "Unknown document"
        )
        page = document.metadata.get("page", "Unknown page")
        print(f"Title: {title}")
        print(f"Page: {page}")

        print(f"\n{document.page_content[:500]}")
