from __future__ import annotations

import re
from collections import defaultdict

from langchain_core.documents import Document as LangChainDocument
from sqlalchemy import bindparam, text

from nasa_rag.catalog import CatalogRepository
from nasa_rag.db import Database
from nasa_rag.retrieval import RetrievedChunk


RETRIEVAL_MODES = {"dense", "keyword", "hybrid"}


class LexicalIndex:
    def __init__(self, database: Database):
        self.database = database

    def search(
        self,
        question: str,
        allowed_version_ids: list[str],
        k: int,
        *,
        filenames: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be positive")
        if not allowed_version_ids:
            return []
        tokens = re.findall(r"[^\W_]+", question, flags=re.UNICODE)[:32]
        if not tokens:
            return []
        match_query = " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        filename_filter = ""
        parameters = {
            "query": match_query,
            "versions": sorted(set(allowed_version_ids)),
            "limit": k,
        }
        statement = """
            SELECT c.id, c.document_id, c.version_id, c.filename,
                   c.page, c.text, bm25(chunk_fts) AS score
            FROM chunk_fts
            JOIN chunks AS c ON c.id = chunk_fts.chunk_id
            JOIN documents AS d ON d.id = c.document_id
            WHERE chunk_fts MATCH :query
              AND chunk_fts.version_id IN :versions
              AND c.section = 'content'
              AND d.deleted_at IS NULL
              AND d.status = 'ready'
              AND d.active_version_id = c.version_id
        """
        bind_parameters = [bindparam("versions", expanding=True)]
        if filenames is not None:
            if not filenames:
                return []
            statement += " AND c.filename IN :filenames"
            parameters["filenames"] = sorted(set(filenames))
            bind_parameters.append(bindparam("filenames", expanding=True))
        statement += " ORDER BY score ASC, c.id ASC LIMIT :limit"
        query = text(statement).bindparams(*bind_parameters)
        with self.database.session() as session:
            rows = session.execute(query, parameters).mappings().all()
        return [
            RetrievedChunk(
                chunk_id=row["id"],
                document_id=row["document_id"],
                version_id=row["version_id"],
                filename=row["filename"],
                page=int(row["page"]),
                text=row["text"],
                distance=float(row["score"]),
                retrieval_source="keyword",
            )
            for row in rows
        ]

    def rebuild_from_canonical_chunks(self) -> int:
        with self.database.session() as session:
            count = int(session.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one())
            session.execute(text("DELETE FROM chunk_fts"))
            session.execute(
                text(
                    """
                    INSERT INTO chunk_fts(
                        chunk_id, document_id, version_id, filename, page, section, text
                    )
                    SELECT id, document_id, version_id, filename, page, section, text
                    FROM chunks
                    """
                )
            )
            return count


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, tuple[int, int]] = {}
    for list_index, ranked in enumerate(ranked_lists):
        for rank, chunk_id in enumerate(dict.fromkeys(ranked), start=1):
            scores[chunk_id] += 1 / (rrf_k + rank)
            first_seen.setdefault(chunk_id, (rank, list_index))
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], first_seen[item[0]], item[0]),
    )


class HybridRetriever:
    def __init__(
        self,
        database: Database,
        dense_index,
        lexical_index: LexicalIndex,
        *,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ):
        self.database = database
        self.dense_index = dense_index
        self.lexical_index = lexical_index
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def search(
        self,
        user_id: str,
        question: str,
        *,
        k: int = 4,
        mode: str = "hybrid",
        document_ids: list[str] | None = None,
        filenames: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if k < 1:
            raise ValueError("k must be positive")
        with self.database.session() as session:
            catalog = CatalogRepository(session)
            if filenames is not None:
                named_document_ids = {
                    document.id
                    for document in catalog.list_accessible_documents(user_id)
                    if document.name in set(filenames)
                }
                document_ids = (
                    sorted(named_document_ids)
                    if document_ids is None
                    else sorted(set(document_ids) & named_document_ids)
                )
            allowed_versions = catalog.allowed_active_version_ids(user_id, document_ids)
        if not allowed_versions:
            return []
        candidate_k = max(k, k * self.candidate_multiplier)
        dense = (
            self.dense_index.search(question, allowed_versions, candidate_k)
            if mode in {"dense", "hybrid"}
            else []
        )
        keyword = (
            self.lexical_index.search(
                question,
                allowed_versions,
                candidate_k,
                filenames=filenames,
            )
            if mode in {"keyword", "hybrid"}
            else []
        )
        if mode == "dense":
            ranked_ids = [item.chunk_id for item in dense]
            fused_scores = {chunk_id: 1 / (rank + 1) for rank, chunk_id in enumerate(ranked_ids)}
        elif mode == "keyword":
            ranked_ids = [item.chunk_id for item in keyword]
            fused_scores = {chunk_id: 1 / (rank + 1) for rank, chunk_id in enumerate(ranked_ids)}
        else:
            fused = reciprocal_rank_fusion(
                [
                    [item.chunk_id for item in dense],
                    [item.chunk_id for item in keyword],
                ],
                rrf_k=self.rrf_k,
            )
            ranked_ids = [chunk_id for chunk_id, _score in fused]
            fused_scores = dict(fused)
        with self.database.session() as session:
            authoritative = CatalogRepository(session).authorized_active_chunks(
                user_id,
                ranked_ids,
                document_ids=document_ids,
                filenames=filenames,
            )
        by_id = {chunk.id: chunk for chunk in authoritative}
        source_by_id = {item.chunk_id: item.retrieval_source for item in dense + keyword}
        results = []
        for chunk_id in ranked_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    version_id=chunk.version_id,
                    filename=chunk.filename,
                    page=chunk.page,
                    text=chunk.text,
                    distance=-fused_scores[chunk_id],
                    retrieval_source=("hybrid" if mode == "hybrid" else source_by_id.get(chunk_id, mode)),
                    rank_score=fused_scores[chunk_id],
                )
            )
            if len(results) == k:
                break
        return results

    @staticmethod
    def as_langchain_results(results: list[RetrievedChunk]):
        return [
            (
                LangChainDocument(
                    page_content=item.text,
                    metadata={
                        "chunk_id": item.chunk_id,
                        "document_id": item.document_id,
                        "version_id": item.version_id,
                        "filename": item.filename,
                        "page": item.page,
                        "retrieval_source": item.retrieval_source,
                    },
                ),
                item.distance,
            )
            for item in results
        ]
