from __future__ import annotations

from sqlalchemy.orm import Session

from nasa_rag.catalog import CatalogRepository
from nasa_rag.models import User
from nasa_rag.security import verify_password


class AuthenticationError(ValueError):
    pass


def authenticate_user(session: Session, username: str, password: str) -> User:
    user = CatalogRepository(session).get_user_by_username(username)
    if (
        user is None
        or not user.is_active
        or not verify_password(password, user.password_hash)
    ):
        raise AuthenticationError("Invalid username or password")
    return user
