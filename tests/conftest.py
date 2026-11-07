import os
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient


os.environ.setdefault(
    "NASA_RAG_JWT_SECRET",
    "tests-only-secret-that-is-at-least-thirty-two-bytes",
)


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    path = tmp_path / "mars-fixture.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Perseverance observed layered rocks in Jezero crater.",
    )
    document.save(path)
    document.close()
    return path


@pytest.fixture
def authenticated_api(tmp_path: Path):
    from api import create_app
    from nasa_rag.catalog import CatalogRepository
    from nasa_rag.db import Database
    from nasa_rag.security import TokenService, hash_password

    database = Database(f"sqlite+pysqlite:///{tmp_path / 'api.db'}")
    database.create_schema()
    with database.session() as session:
        user = CatalogRepository(session).create_user(
            "test-owner", hash_password("test-owner-password")
        )
    tokens = TokenService("t" * 32)
    upload_dir = tmp_path / "uploads"
    app = create_app(
        database=database,
        token_service=tokens,
        upload_dir=upload_dir,
    )
    return {
        "app": app,
        "client": TestClient(app),
        "database": database,
        "headers": {"Authorization": f"Bearer {tokens.issue(user.id)}"},
        "upload_dir": upload_dir,
        "user": user,
    }
