from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from langchain_core.embeddings import Embeddings
from sqlalchemy import inspect

from nasa_rag.catalog import CatalogRepository, DocumentAccessError
from nasa_rag.cli import create_user
from nasa_rag.db import Database
from nasa_rag.retrieval import ChromaDenseIndex, IndexedChunk
from nasa_rag.security import InvalidTokenError, TokenService, hash_password, verify_password


class DeterministicEmbeddings(Embeddings):
    def _vector(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float(lowered.count("alice")),
            float(lowered.count("bob")),
            float(lowered.count("mars")),
            1.0,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def database(tmp_path) -> Database:
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'catalog.db'}")
    database.create_schema()
    return database


def _seed_private_documents(database: Database):
    with database.session() as session:
        catalog = CatalogRepository(session)
        alice = catalog.create_user("alice", hash_password("alice-password"))
        bob = catalog.create_user("bob", hash_password("bob-password"))
        alice_document = catalog.create_document(alice.id, "same.pdf")
        bob_document = catalog.create_document(bob.id, "same.pdf")
        alice_version = catalog.create_version(
            alice_document.id,
            content_sha256="a" * 64,
            filename="same.pdf",
            storage_path="private/alice.pdf",
        )
        bob_version = catalog.create_version(
            bob_document.id,
            content_sha256="b" * 64,
            filename="same.pdf",
            storage_path="private/bob.pdf",
        )
        catalog.activate_version(alice_document.id, alice_version.id)
        catalog.activate_version(bob_document.id, bob_version.id)
        return alice, bob, alice_document, bob_document, alice_version, bob_version


def test_passwords_use_argon2_and_never_store_plaintext():
    encoded = hash_password("correct horse battery staple")

    assert encoded.startswith("$argon2id$")
    assert "correct horse battery staple" not in encoded
    assert verify_password("correct horse battery staple", encoded) is True
    assert verify_password("wrong", encoded) is False


def test_access_tokens_reject_wrong_secret_and_expired_tokens():
    issuer = TokenService("a" * 32, lifetime=timedelta(minutes=15))
    verifier = TokenService("b" * 32, lifetime=timedelta(minutes=15))

    token = issuer.issue("user-123")
    assert issuer.verify(token) == "user-123"
    with pytest.raises(InvalidTokenError):
        verifier.verify(token)

    expired = TokenService("a" * 32, lifetime=timedelta(seconds=-1)).issue(
        "user-123"
    )
    with pytest.raises(InvalidTokenError):
        issuer.verify(expired)


def test_initial_migration_creates_empty_identity_and_acl_schema(tmp_path: Path):
    database_path = tmp_path / "migrated.db"
    config = Config("alembic.ini")
    config.set_main_option(
        "sqlalchemy.url", f"sqlite+pysqlite:///{database_path.as_posix()}"
    )

    command.upgrade(config, "head")

    database = Database(f"sqlite+pysqlite:///{database_path.as_posix()}")
    assert {
        "users",
        "documents",
        "document_versions",
        "document_permissions",
        "chunks",
        "ingestion_jobs",
        "alembic_version",
    }.issubset(inspect(database.engine).get_table_names())
    version_columns = {
        column["name"]
        for column in inspect(database.engine).get_columns("document_versions")
    }
    assert {"based_on_version_id", "derived_cleaned_at"}.issubset(version_columns)
    with database.session() as session:
        assert CatalogRepository(session).get_user_by_username("admin") is None


def test_bootstrap_user_creation_requires_explicit_credentials(database):
    user_id = create_user(database, "bootstrap-owner", "long-local-password")

    with database.session() as session:
        user = CatalogRepository(session).get_user_by_username("bootstrap-owner")
        assert user is not None
        assert user.id == user_id
        assert verify_password("long-local-password", user.password_hash)


def test_owner_visibility_reader_grant_and_revocation_are_deny_by_default(database):
    alice, bob, alice_document, bob_document, alice_version, bob_version = (
        _seed_private_documents(database)
    )

    with database.session() as session:
        catalog = CatalogRepository(session)
        assert [item.id for item in catalog.list_accessible_documents(alice.id)] == [
            alice_document.id
        ]
        assert [item.id for item in catalog.list_accessible_documents(bob.id)] == [
            bob_document.id
        ]
        with pytest.raises(DocumentAccessError):
            catalog.require_document(bob.id, alice_document.id)

        catalog.grant_reader(alice.id, alice_document.id, bob.id)
        assert set(catalog.allowed_active_version_ids(bob.id)) == {
            alice_version.id,
            bob_version.id,
        }

        catalog.revoke_reader(alice.id, alice_document.id, bob.id)
        assert catalog.allowed_active_version_ids(bob.id) == [bob_version.id]


def test_dense_search_filters_private_versions_before_candidate_retrieval(
    database, tmp_path
):
    alice, bob, _, _, alice_version, bob_version = _seed_private_documents(database)
    index = ChromaDenseIndex(tmp_path / "chroma", DeterministicEmbeddings())
    index.upsert(
        [
            IndexedChunk(
                chunk_id="alice-chunk",
                document_id=alice_version.document_id,
                version_id=alice_version.id,
                filename="same.pdf",
                page=1,
                text="alice private landing site evidence",
            ),
            IndexedChunk(
                chunk_id="bob-chunk",
                document_id=bob_version.document_id,
                version_id=bob_version.id,
                filename="same.pdf",
                page=1,
                text="bob public-facing mars evidence",
            ),
        ]
    )

    with database.session() as session:
        allowed = CatalogRepository(session).allowed_active_version_ids(bob.id)
    results = index.search("alice", allowed_version_ids=allowed, k=10)

    assert [result.chunk_id for result in results] == ["bob-chunk"]
    assert all(result.version_id == bob_version.id for result in results)


def test_shared_reader_search_visibility_disappears_immediately_after_revocation(
    database, tmp_path
):
    alice, bob, alice_document, _, alice_version, bob_version = (
        _seed_private_documents(database)
    )
    index = ChromaDenseIndex(tmp_path / "chroma", DeterministicEmbeddings())
    index.upsert(
        [
            IndexedChunk(
                chunk_id="alice-chunk",
                document_id=alice_document.id,
                version_id=alice_version.id,
                filename="same.pdf",
                page=1,
                text="alice mars sample",
            )
        ]
    )

    with database.session() as session:
        catalog = CatalogRepository(session)
        catalog.grant_reader(alice.id, alice_document.id, bob.id)
        shared = catalog.allowed_active_version_ids(bob.id)
    assert [item.chunk_id for item in index.search("alice", shared, 5)] == [
        "alice-chunk"
    ]

    with database.session() as session:
        catalog = CatalogRepository(session)
        catalog.revoke_reader(alice.id, alice_document.id, bob.id)
        revoked = catalog.allowed_active_version_ids(bob.id)
    assert revoked == [bob_version.id]
    assert index.search("alice", revoked, 5) == []
