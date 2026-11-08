import os
from pathlib import Path

import chromadb
import requests
from sqlalchemy import inspect, text

OLLAMA_TAGS_URL = (
    os.getenv("NASA_RAG_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    + "/api/tags"
)
REQUIRED_MODELS = {"nomic-embed-text", "qwen3:1.7b"}
COLLECTION = "nasa_mars_uploads"


def _check_vector_store(vector_store_path: Path) -> dict:
    if not vector_store_path.is_dir():
        return {"status": "missing"}
    try:
        client = chromadb.PersistentClient(path=str(vector_store_path))
        collection = client.get_collection(COLLECTION)
        sample = collection.get(limit=1, include=["metadatas"])
        if not sample.get("ids"):
            return {"status": "empty"}
        metadata = (sample.get("metadatas") or [None])[0] or {}
        required_metadata = {
            "document_id",
            "version_id",
            "filename",
            "section",
        }
        if not required_metadata.issubset(metadata):
            return {"status": "invalid_metadata"}
        return {"status": "ok", "chunks": collection.count()}
    except Exception as error:
        return {"status": "unavailable", "error_type": type(error).__name__}


def _check_catalog(engine) -> dict:
    try:
        required_tables = {
            "users",
            "documents",
            "document_versions",
            "document_permissions",
            "chunks",
            "ingestion_jobs",
        }
        if not required_tables.issubset(inspect(engine).get_table_names()):
            return {"status": "migration_required"}
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            if engine.dialect.name == "sqlite":
                fts = connection.execute(
                    text(
                        "SELECT 1 FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'chunk_fts'"
                    )
                ).first()
                if fts is None:
                    return {"status": "migration_required"}
        return {"status": "ok"}
    except Exception as error:
        return {"status": "unavailable", "error_type": type(error).__name__}


def check_readiness(
    *,
    vector_store_path: Path = Path("data/chroma"),
    database_engine=None,
) -> dict:
    checks = {
        "catalog": (
            _check_catalog(database_engine)
            if database_engine is not None
            else {"status": "unknown"}
        ),
        "vector_store": _check_vector_store(vector_store_path),
        "ollama": {"status": "unknown"},
        "models": {"status": "unknown"},
    }
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=(1, 2))
        response.raise_for_status()
        checks["ollama"] = {"status": "ok"}
        installed = {
            item.get("name", "").removesuffix(":latest")
            for item in response.json().get("models", [])
        }
        missing = sorted(REQUIRED_MODELS - installed)
        checks["models"] = (
            {"status": "ok"}
            if not missing
            else {"status": "missing", "models": missing}
        )
    except (requests.RequestException, ValueError):
        checks["ollama"] = {"status": "unavailable"}

    ready = all(check["status"] == "ok" for check in checks.values())
    return {"ready": ready, "checks": checks}
