from langchain_chroma import Chroma
from retrieval_embeddings import NomicRetrievalEmbeddings


def list_documents() -> list[dict]:

    store = Chroma(
        collection_name="nasa_mars_uploads",
        embedding_function=NomicRetrievalEmbeddings(),
        persist_directory="data/chroma",
    )

    data = store.get(include=["metadatas"])

    documents = {}

    for metadata in data["metadatas"]:
        if not metadata:
            continue

        file_hash = metadata.get("sha256")
        filename = metadata.get("filename")

        if not file_hash or not filename:
            continue

        documents[file_hash] = {
            "sha256": file_hash,
            "filename": filename,
        }

    return sorted(
        documents.values(),
        key=lambda doc: doc["filename"].lower(),
    )


if __name__ == "__main__":
    for document in list_documents():
        print(document["filename"])
        print("SHA-256:", document["sha256"])
        print()
