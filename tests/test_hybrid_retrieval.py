from __future__ import annotations

from unittest.mock import Mock, patch

from sqlalchemy import text

import rag_nasa
from evaluation.retrieval_ab import compare_retrieval_modes
from nasa_rag.catalog import CatalogRepository
from nasa_rag.db import Database
from nasa_rag.hybrid import (
    HybridRetriever,
    LexicalIndex,
    reciprocal_rank_fusion,
)
from nasa_rag.models import Chunk
from nasa_rag.retrieval import RetrievedChunk
from nasa_rag.security import hash_password


class ScriptedDenseIndex:
    def __init__(self, results=()):
        self.results = list(results)
        self.allowed_calls = []

    def search(self, question, allowed_version_ids, k):
        self.allowed_calls.append(set(allowed_version_ids))
        return self.results[:k]


def _chunk(chunk_id, document_id, version_id, text, filename="same.pdf"):
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        version_id=version_id,
        ordinal=1,
        page=1,
        section="content",
        filename=filename,
        text=text,
    )


def _retrieved(chunk):
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        version_id=chunk.version_id,
        filename=chunk.filename,
        page=chunk.page,
        text=chunk.text,
        distance=0.1,
    )


def _seed(database):
    with database.session() as session:
        catalog = CatalogRepository(session)
        alice = catalog.create_user("alice-hybrid", hash_password("alice-password"))
        bob = catalog.create_user("bob-hybrid", hash_password("bob-password"))

        alice_doc = catalog.create_document(alice.id, "same.pdf")
        alice_old = catalog.create_version(
            alice_doc.id,
            content_sha256="a" * 64,
            filename="same.pdf",
            storage_path="private/alice-old.pdf",
        )
        old_chunk = _chunk(
            "alice-old", alice_doc.id, alice_old.id, "SHERLOC stale private evidence"
        )
        catalog.replace_chunks(alice_old.id, [old_chunk])
        catalog.activate_version(alice_doc.id, alice_old.id)
        alice_active = catalog.create_version(
            alice_doc.id,
            content_sha256="b" * 64,
            filename="same.pdf",
            storage_path="private/alice-active.pdf",
            based_on_version_id=alice_old.id,
        )
        alice_chunk = _chunk(
            "alice-active",
            alice_doc.id,
            alice_active.id,
            "SHERLOC examined the Maaz target",
        )
        catalog.replace_chunks(alice_active.id, [alice_chunk])
        catalog.promote_version(
            alice_doc.id,
            alice_active.id,
            expected_active_version_id=alice_old.id,
        )

        bob_doc = catalog.create_document(bob.id, "same.pdf")
        bob_active = catalog.create_version(
            bob_doc.id,
            content_sha256="c" * 64,
            filename="same.pdf",
            storage_path="private/bob.pdf",
        )
        bob_chunk = _chunk(
            "bob-active",
            bob_doc.id,
            bob_active.id,
            "SHERLOC private Bob identifier",
        )
        catalog.replace_chunks(bob_active.id, [bob_chunk])
        catalog.activate_version(bob_doc.id, bob_active.id)
        return {
            "alice": alice,
            "bob": bob,
            "alice_doc": alice_doc,
            "alice_old": alice_old,
            "alice_active": alice_active,
            "alice_chunk": alice_chunk,
            "old_chunk": old_chunk,
            "bob_doc": bob_doc,
            "bob_active": bob_active,
            "bob_chunk": bob_chunk,
        }


def test_bm25_prefilters_private_and_inactive_versions_with_equal_names(tmp_path):
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'hybrid.db'}")
    database.create_schema()
    seeded = _seed(database)
    lexical = LexicalIndex(database)
    with database.session() as session:
        allowed = CatalogRepository(session).allowed_active_version_ids(
            seeded["alice"].id
        )

    results = lexical.search("SHERLOC Maaz", allowed, k=10)

    assert [result.chunk_id for result in results] == ["alice-active"]
    assert all("Bob" not in result.text and "stale" not in result.text for result in results)


def test_rrf_is_deterministic_deduplicated_and_does_not_add_raw_scores():
    fused = reciprocal_rank_fusion(
        [["dense-a", "shared"], ["shared", "keyword-b"]],
        rrf_k=60,
    )

    assert [chunk_id for chunk_id, _score in fused] == [
        "shared",
        "dense-a",
        "keyword-b",
    ]
    assert len({chunk_id for chunk_id, _score in fused}) == 3


def test_keyword_rebuild_uses_canonical_chunks_without_dense_embeddings(tmp_path):
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'rebuild.db'}")
    database.create_schema()
    seeded = _seed(database)
    lexical = LexicalIndex(database)
    with database.session() as session:
        session.execute(text("DELETE FROM chunk_fts"))

    assert lexical.rebuild_from_canonical_chunks() == 3
    with database.session() as session:
        allowed = CatalogRepository(session).allowed_active_version_ids(
            seeded["alice"].id
        )
    assert [item.chunk_id for item in lexical.search("Maaz", allowed, 4)] == [
        "alice-active"
    ]


def test_dense_keyword_and_hybrid_modes_are_independent_and_revalidated(tmp_path):
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'modes.db'}")
    database.create_schema()
    seeded = _seed(database)
    dense = ScriptedDenseIndex(
        [
            _retrieved(seeded["bob_chunk"]),
            _retrieved(seeded["alice_chunk"]),
        ]
    )
    retriever = HybridRetriever(database, dense, LexicalIndex(database), rrf_k=60)

    dense_results = retriever.search(
        seeded["alice"].id, "weathered target", k=5, mode="dense"
    )
    keyword_results = retriever.search(
        seeded["alice"].id, "SHERLOC", k=5, mode="keyword"
    )
    hybrid_results = retriever.search(
        seeded["alice"].id, "SHERLOC", k=5, mode="hybrid"
    )

    assert [item.chunk_id for item in dense_results] == ["alice-active"]
    assert [item.chunk_id for item in keyword_results] == ["alice-active"]
    assert [item.chunk_id for item in hybrid_results] == ["alice-active"]
    assert dense.allowed_calls == [
        {seeded["alice_active"].id},
        {seeded["alice_active"].id},
    ]


def test_private_dense_candidate_never_enters_model_prompt(tmp_path):
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'prompt.db'}")
    database.create_schema()
    seeded = _seed(database)
    dense = ScriptedDenseIndex(
        [_retrieved(seeded["bob_chunk"]), _retrieved(seeded["alice_chunk"])]
    )
    retriever = HybridRetriever(database, dense, LexicalIndex(database))
    model = Mock()
    model.invoke.return_value = Mock(content="Maaz was examined [1].")

    def authorised_search(question):
        return retriever.as_langchain_results(
            retriever.search(
                seeded["alice"].id, question, k=4, mode="hybrid"
            )
        )

    with patch.object(rag_nasa, "llm", model):
        rag_nasa.answer_question("What did SHERLOC examine?", search_function=authorised_search)

    prompt = model.invoke.call_args.args[0][1][1]
    assert "Maaz" in prompt
    assert "private Bob" not in prompt
    assert "stale private" not in prompt


def test_retrieval_ab_uses_identical_cases_and_reports_recall_and_percentiles():
    cases = [
        {"id": "one", "question": "SHERLOC", "relevant_ids": ["a"]},
        {"id": "two", "question": "altered", "relevant_ids": ["b"]},
    ]

    def retrieve(mode, case, _k):
        ids = ["a"] if case["id"] == "one" else (["b"] if mode == "hybrid" else [])
        return {
            "retrieved_ids": ids,
            "answer_correct": bool(ids),
            "citation_supported": bool(ids),
        }

    result = compare_retrieval_modes(cases, retrieve, modes=("dense", "hybrid"), k=4)

    assert result["sample_ids"] == ["one", "two"]
    assert result["modes"]["dense"]["recall_at_k"] == 0.5
    assert result["modes"]["hybrid"]["recall_at_k"] == 1.0
    assert result["modes"]["dense"]["answer_correctness"] == 0.5
    assert result["modes"]["hybrid"]["citation_support"] == 1.0
    assert result["modes"]["hybrid"]["latency_ms"]["p50"] >= 0
    assert result["modes"]["hybrid"]["latency_ms"]["p95"] >= 0
