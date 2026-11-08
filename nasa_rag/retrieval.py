from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    version_id: str
    filename: str
    page: int
    text: str
    section: str = "content"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    version_id: str
    filename: str
    page: int
    text: str
    distance: float
    retrieval_source: str = "dense"
    rank_score: float | None = None


class ChromaDenseIndex:
    def __init__(
        self,
        path: Path,
        embeddings: Embeddings,
        *,
        collection_name: str = "nasa_mars_uploads",
    ):
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(path),
        )

    def upsert(self, chunks: list[IndexedChunk]) -> None:
        if not chunks:
            return
        documents = [
            LangChainDocument(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "version_id": chunk.version_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "section": chunk.section,
                },
            )
            for chunk in chunks
        ]
        self.store.add_documents(
            documents=documents,
            ids=[chunk.chunk_id for chunk in chunks],
        )

    def search(
        self,
        question: str,
        allowed_version_ids: list[str],
        k: int,
    ) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        if not allowed_version_ids:
            return []
        allowed = set(allowed_version_ids)
        metadata_filter = {
            "$and": [
                {"section": {"$eq": "content"}},
                {"version_id": {"$in": sorted(allowed)}},
            ]
        }
        candidates = self.store.similarity_search_with_score(
            question,
            k=k,
            filter=metadata_filter,
        )
        results: list[RetrievedChunk] = []
        for document, distance in candidates:
            metadata = document.metadata
            version_id = str(metadata.get("version_id", ""))
            if version_id not in allowed:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=str(metadata["chunk_id"]),
                    document_id=str(metadata["document_id"]),
                    version_id=version_id,
                    filename=str(metadata["filename"]),
                    page=int(metadata["page"]),
                    text=document.page_content,
                    distance=float(distance),
                )
            )
        return results

    def delete_version(self, version_id: str) -> None:
        data = self.store.get(where={"version_id": version_id}, include=[])
        if data["ids"]:
            self.store.delete(ids=list(data["ids"]))

    def version_chunk_ids(self, version_id: str) -> set[str]:
        data = self.store.get(where={"version_id": version_id}, include=[])
        return set(data["ids"])

    def verify(self, chunks: list[IndexedChunk]) -> bool:
        if not chunks:
            return False
        expected = {chunk.chunk_id: chunk for chunk in chunks}
        data = self.store.get(
            ids=list(expected),
            include=["documents", "metadatas"],
        )
        if set(data["ids"]) != set(expected):
            return False
        for chunk_id, text, metadata in zip(
            data["ids"], data["documents"], data["metadatas"], strict=True
        ):
            chunk = expected[chunk_id]
            if (
                text != chunk.text
                or metadata.get("document_id") != chunk.document_id
                or metadata.get("version_id") != chunk.version_id
                or metadata.get("chunk_id") != chunk.chunk_id
                or int(metadata.get("page", -1)) != chunk.page
                or metadata.get("section") != chunk.section
            ):
                return False
        return True
