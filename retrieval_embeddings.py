import os

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


class NomicRetrievalEmbeddings(Embeddings):
    def __init__(self, base_url: str | None = None):
        self.model = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=(
                base_url
                or os.getenv("NASA_RAG_OLLAMA_BASE_URL", "http://localhost:11434")
            ),
            client_kwargs={"timeout": 30.0},
        )

    def embed_documents(self, texts):
        prefixed = [f"search_document: {t}" for t in texts]
        return self.model.embed_documents(prefixed)

    def embed_query(self, text):
        return self.model.embed_query(f"search_query: {text}")
