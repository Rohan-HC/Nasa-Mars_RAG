from pathlib import Path

import chromadb
import requests


OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
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
        required_metadata = {"sha256", "filename", "section"}
        if not required_metadata.issubset(metadata):
            return {"status": "invalid_metadata"}
        return {"status": "ok", "chunks": collection.count()}
    except Exception as error:
        return {"status": "unavailable", "error_type": type(error).__name__}


def check_readiness(
    *,
    vector_store_path: Path = Path("data/chroma"),
) -> dict:
    checks = {
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
