from pathlib import Path

import requests


OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
REQUIRED_MODELS = {"nomic-embed-text", "qwen3:1.7b"}


def check_readiness(
    *,
    vector_store_path: Path = Path("data/chroma"),
) -> dict:
    checks = {
        "vector_store": {
            "status": "ok" if vector_store_path.exists() else "missing"
        },
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
