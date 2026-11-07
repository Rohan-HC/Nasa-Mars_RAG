from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
from pwdlib import PasswordHash


_password_hash = PasswordHash.recommended()


class InvalidTokenError(ValueError):
    pass


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters")
    return _password_hash.hash(password)


def verify_password(password: str, encoded: str) -> bool:
    try:
        return _password_hash.verify(password, encoded)
    except (ValueError, TypeError):
        return False


class TokenService:
    def __init__(
        self,
        secret: str,
        *,
        lifetime: timedelta = timedelta(minutes=15),
        issuer: str = "nasa-mars-rag",
    ):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("JWT secret must contain at least 32 bytes")
        self.secret = secret
        self.lifetime = lifetime
        self.issuer = issuer

    def issue(self, user_id: str) -> str:
        now = datetime.now(UTC)
        return jwt.encode(
            {
                "sub": user_id,
                "iss": self.issuer,
                "iat": now,
                "exp": now + self.lifetime,
                "jti": str(uuid4()),
                "type": "access",
            },
            self.secret,
            algorithm="HS256",
        )

    def verify(self, token: str) -> str:
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=["HS256"],
                issuer=self.issuer,
                options={"require": ["sub", "iss", "iat", "exp", "jti"]},
            )
        except jwt.PyJWTError as error:
            raise InvalidTokenError("Invalid or expired access token") from error
        if payload.get("type") != "access" or not isinstance(payload.get("sub"), str):
            raise InvalidTokenError("Invalid or expired access token")
        return payload["sub"]
