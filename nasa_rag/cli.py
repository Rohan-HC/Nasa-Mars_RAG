from __future__ import annotations

import argparse
import getpass

from sqlalchemy.exc import IntegrityError

from nasa_rag.catalog import CatalogRepository
from nasa_rag.config import Settings
from nasa_rag.db import Database
from nasa_rag.security import hash_password


def create_user(database: Database, username: str, password: str) -> str:
    try:
        with database.session() as session:
            user = CatalogRepository(session).create_user(
                username,
                hash_password(password),
            )
            return user.id
    except IntegrityError as error:
        raise ValueError("Username already exists") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="NASA Mars RAG administration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create-user", description="Create an explicit local user"
    )
    create.add_argument("--username", required=True)
    args = parser.parse_args()
    if args.command == "create-user":
        password = getpass.getpass("Password (minimum 12 characters): ")
        confirmation = getpass.getpass("Confirm password: ")
        if password != confirmation:
            parser.error("Passwords do not match")
        settings = Settings.from_env()
        user_id = create_user(Database(settings.database_url), args.username, password)
        print(f"Created user {args.username.casefold()} ({user_id})")


if __name__ == "__main__":
    main()
