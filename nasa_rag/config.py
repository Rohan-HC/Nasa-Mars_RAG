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
        return cls(
            jwt_secret=secret,
            database_url=os.getenv(
                "NASA_RAG_DATABASE_URL", "sqlite+pysqlite:///data/catalog.db"
            ),
            chroma_path=Path(os.getenv("NASA_RAG_CHROMA_PATH", "data/chroma")),
            upload_dir=Path(os.getenv("NASA_RAG_UPLOAD_DIR", "data/uploads")),
            access_token_minutes=token_minutes,
        )
