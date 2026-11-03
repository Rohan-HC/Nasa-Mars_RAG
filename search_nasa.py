from pathlib import Path

from langchain_chroma import Chroma
from retrieval_embeddings import NomicRetrievalEmbeddings


DB_PATH = "data/chroma"
COLLECTION = "nasa_mars_uploads"


def search_nasa(question: str):
    if not question.strip():
        raise ValueError("Question must not be empty")

    if not Path(DB_PATH).exists():
        raise FileNotFoundError("NASA vector database not found")

    embeddings = NomicRetrievalEmbeddings()

    vector_store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    results = vector_store.similarity_search_with_score(
        question, k=3, filter={"section": "content"}
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
