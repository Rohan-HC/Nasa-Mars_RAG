from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import create_app
from nasa_rag.catalog import CatalogRepository
from nasa_rag.db import Database
from nasa_rag.indexing import IndexingService
from nasa_rag.rate_limit import SlidingWindowRateLimiter
from nasa_rag.security import TokenService, hash_password


@pytest.fixture
def secure_api(tmp_path: Path):
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'catalog.db'}")
    database.create_schema()
    with database.session() as session:
        catalog = CatalogRepository(session)
        alice = catalog.create_user("alice", hash_password("alice-password"))
        bob = catalog.create_user("bob", hash_password("bob-password"))
        alice_document = catalog.create_document(alice.id, "alice-private.pdf")
        bob_document = catalog.create_document(bob.id, "bob-private.pdf")
        alice_version = catalog.create_version(
            alice_document.id,
            content_sha256="a" * 64,
            filename="alice-private.pdf",
            storage_path="private/alice.pdf",
        )
        bob_version = catalog.create_version(
            bob_document.id,
            content_sha256="b" * 64,
            filename="bob-private.pdf",
            storage_path="private/bob.pdf",
        )
        catalog.activate_version(alice_document.id, alice_version.id)
        catalog.activate_version(bob_document.id, bob_version.id)

    def answer_function(*, question: str, allowed_version_ids: list[str]):
        return {
            "answer": "Authorised evidence [1].",
            "retrieved": [
                {"label": 1, "version_id": version_id, "text": "safe evidence"}
                for version_id in allowed_version_ids
            ],
            "validation": {"citation_valid": True, "passed": True, "errors": []},
        }

    def comparison_function(**_kwargs):
        return {
            "answer": "Comparison [S1, S2].",
            "retrieved": [],
            "validation": {"citation_valid": True, "passed": True, "errors": []},
        }

    tokens = TokenService("s" * 32)
    app = create_app(
        database=database,
        token_service=tokens,
        answer_function=answer_function,
        comparison_function=comparison_function,
        upload_dir=tmp_path / "uploads",
    )
    client = TestClient(app)

    def login(username: str, password: str) -> dict[str, str]:
        response = client.post(
            "/auth/token", json={"username": username, "password": password}
        )
        assert response.status_code == 200
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    return {
        "database": database,
        "client": client,
        "alice": alice,
        "bob": bob,
        "alice_document": alice_document,
        "bob_document": bob_document,
        "alice_version": alice_version,
        "bob_version": bob_version,
        "alice_headers": login("alice", "alice-password"),
        "bob_headers": login("bob", "bob-password"),
    }


def test_document_routes_require_authentication(secure_api):
    client = secure_api["client"]

    assert client.get("/documents").status_code == 401
    assert client.post("/ask", json={"question": "Mars?"}).status_code == 401
    assert (
        client.post(
            "/compare/documents",
            json={
                "document_a": secure_api["alice_document"].id,
                "document_b": secure_api["bob_document"].id,
                "question": "Compare",
            },
        ).status_code
        == 401
    )
    assert client.post("/documents/upload").status_code == 401


def test_login_failure_is_generic_and_does_not_enumerate_users(secure_api):
    client = secure_api["client"]
    missing = client.post(
        "/auth/token", json={"username": "nobody", "password": "wrong-password"}
    )
    wrong = client.post(
        "/auth/token", json={"username": "alice", "password": "wrong-password"}
    )

    assert missing.status_code == wrong.status_code == 401
    assert missing.json()["error"]["message"] == wrong.json()["error"]["message"]


def test_campaign_comparison_obeys_authenticated_user_rate_limit(secure_api):
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
    assert limiter.allow(secure_api["alice"].id)
    secure_api["client"].app.state.user_limiter = limiter

    response = secure_api["client"].post(
        "/compare/campaigns",
        headers=secure_api["alice_headers"],
        json={"question": "Compare the campaigns"},
    )

    assert response.status_code == 429


def test_repeated_login_attempts_are_rate_limited(tmp_path):
    database = Database(f"sqlite+pysqlite:///{tmp_path / 'rate.db'}")
    database.create_schema()
    app = create_app(
        database=database,
        token_service=TokenService("r" * 32),
        login_limiter=SlidingWindowRateLimiter(limit=1, window_seconds=60),
        upload_dir=tmp_path / "uploads",
    )
    client = TestClient(app)

    first = client.post(
        "/auth/token", json={"username": "nobody", "password": "wrong-password"}
    )
    second = client.post(
        "/auth/token", json={"username": "nobody", "password": "wrong-password"}
    )

    assert first.status_code == 401
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"


def test_bob_cannot_list_or_guess_alice_document(secure_api):
    client = secure_api["client"]
    response = client.get("/documents", headers=secure_api["bob_headers"])

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["documents"][0]["document_id"] == secure_api[
        "bob_document"
    ].id
    guessed = client.get(
        f"/documents/{secure_api['alice_document'].id}",
        headers=secure_api["bob_headers"],
    )
    assert guessed.status_code == 404
    assert "alice" not in guessed.text.casefold()
    assert "a" * 64 not in guessed.text


def test_ask_uses_only_server_derived_active_versions(secure_api):
    response = secure_api["client"].post(
        "/ask",
        headers=secure_api["bob_headers"],
        json={"question": "Mars?"},
    )

    assert response.status_code == 200
    assert [item["version_id"] for item in response.json()["retrieved"]] == [
        secure_api["bob_version"].id
    ]


def test_owner_grant_enables_read_and_revocation_removes_it_immediately(secure_api):
    client = secure_api["client"]
    document_id = secure_api["alice_document"].id
    grant = client.put(
        f"/documents/{document_id}/permissions/bob",
        headers=secure_api["alice_headers"],
        json={"role": "reader"},
    )
    assert grant.status_code == 200

    shared = client.post(
        "/ask",
        headers=secure_api["bob_headers"],
        json={"question": "Mars?"},
    )
    assert {item["version_id"] for item in shared.json()["retrieved"]} == {
        secure_api["alice_version"].id,
        secure_api["bob_version"].id,
    }

    revoked = client.delete(
        f"/documents/{document_id}/permissions/bob",
        headers=secure_api["alice_headers"],
    )
    assert revoked.status_code == 204
    after = client.post(
        "/ask",
        headers=secure_api["bob_headers"],
        json={"question": "Mars?"},
    )
    assert [item["version_id"] for item in after.json()["retrieved"]] == [
        secure_api["bob_version"].id
    ]


def test_bob_cannot_compare_with_alice_document_or_change_its_acl(secure_api):
    client = secure_api["client"]
    alice_id = secure_api["alice_document"].id
    bob_id = secure_api["bob_document"].id
    comparison = client.post(
        "/compare/documents",
        headers=secure_api["bob_headers"],
        json={
            "document_a": alice_id,
            "document_b": bob_id,
            "question": "Compare",
        },
    )
    grant = client.put(
        f"/documents/{alice_id}/permissions/alice",
        headers=secure_api["bob_headers"],
        json={"role": "reader"},
    )

    assert comparison.status_code == 404
    assert grant.status_code == 404


def test_bob_cannot_delete_alice_document(secure_api):
    class EmptyExtractor:
        def extract(self, _path):
            raise AssertionError("deletion must not extract content")

    class MemoryIndex:
        def delete_version(self, _version_id):
            return None

        def upsert(self, _chunks):
            raise AssertionError("deletion must not index content")

        def verify(self, _chunks):
            return False

        def version_chunk_ids(self, _version_id):
            return set()

    secure_api["client"].app.state.indexing_service = IndexingService(
        secure_api["database"], EmptyExtractor(), MemoryIndex()
    )

    response = secure_api["client"].delete(
        f"/documents/{secure_api['alice_document'].id}",
        headers=secure_api["bob_headers"],
    )

    assert response.status_code == 404
    with secure_api["database"].session() as session:
        assert CatalogRepository(session).require_document(
            secure_api["alice"].id,
            secure_api["alice_document"].id,
        )
