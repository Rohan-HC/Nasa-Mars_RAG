from collections import Counter

from langchain_chroma import Chroma
from retrieval_embeddings import NomicRetrievalEmbeddings

store = Chroma(
    collection_name="nasa_mars_uploads",
    embedding_function=NomicRetrievalEmbeddings(),
    persist_directory="data/chroma",
)

data = store.get(include=["metadatas"])

counts = Counter(metadata.get("filename", "Unknown") for metadata in data["metadatas"])

print("Total chunks:", len(data["ids"]))
print("\nDocuments:")

for filename, count in counts.items():
    print(f"{filename}: {count} chunks")
