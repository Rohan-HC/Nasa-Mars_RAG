from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    jwt_secret: str
    database_url: str = "sqlite+pysqlite:///data/catalog.db"
    chroma_path: Path = Path("data/chroma")
    upload_dir: Path = Path("data/uploads")
    access_token_minutes: int = 15
    retrieval_mode: str = "hybrid"
    retrieval_k: int = 4
    rrf_k: int = 60
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "Settings":
        secret = os.getenv("NASA_RAG_JWT_SECRET", "")
        if len(secret.encode("utf-8")) < 32:
            raise ConfigurationError(
                "NASA_RAG_JWT_SECRET must contain at least 32 bytes"
            )
        try:
            token_minutes = int(os.getenv("NASA_RAG_ACCESS_TOKEN_MINUTES", "15"))
        except ValueError as error:
            raise ConfigurationError(
                "NASA_RAG_ACCESS_TOKEN_MINUTES must be an integer"
            ) from error
        if not 1 <= token_minutes <= 60:
            raise ConfigurationError(
                "NASA_RAG_ACCESS_TOKEN_MINUTES must be between 1 and 60"
            )
        retrieval_mode = os.getenv("NASA_RAG_RETRIEVAL_MODE", "hybrid")
        if retrieval_mode not in {"dense", "keyword", "hybrid"}:
            raise ConfigurationError(
                "NASA_RAG_RETRIEVAL_MODE must be dense, keyword or hybrid"
            )
        try:
            retrieval_k = int(os.getenv("NASA_RAG_RETRIEVAL_K", "4"))
            rrf_k = int(os.getenv("NASA_RAG_RRF_K", "60"))
        except ValueError as error:
            raise ConfigurationError("Retrieval K settings must be integers") from error
        if not 1 <= retrieval_k <= 20 or not 1 <= rrf_k <= 1000:
            raise ConfigurationError("Retrieval K settings are outside safe bounds")
        return cls(
            jwt_secret=secret,
            database_url=os.getenv(
                "NASA_RAG_DATABASE_URL", "sqlite+pysqlite:///data/catalog.db"
            ),
            chroma_path=Path(os.getenv("NASA_RAG_CHROMA_PATH", "data/chroma")),
            upload_dir=Path(os.getenv("NASA_RAG_UPLOAD_DIR", "data/uploads")),
            access_token_minutes=token_minutes,
            retrieval_mode=retrieval_mode,
            retrieval_k=retrieval_k,
            rrf_k=rrf_k,
            ollama_base_url=os.getenv(
                "NASA_RAG_OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
        )
