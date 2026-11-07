from __future__ import annotations

from sqlalchemy import exists, or_, select
from sqlalchemy.orm import Session

from nasa_rag.models import (
    Document,
    DocumentPermission,
    DocumentVersion,
    User,
    utc_now,
)


class DocumentAccessError(LookupError):
    pass


class CatalogRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_user(self, username: str, password_hash: str) -> User:
        normalized = username.strip().casefold()
        if not normalized or len(normalized) > 64:
            raise ValueError("Username must contain between 1 and 64 characters")
        user = User(username=normalized, password_hash=password_hash)
        self.session.add(user)
        self.session.flush()
        return user

    def get_user_by_username(self, username: str) -> User | None:
        return self.session.scalar(
            select(User).where(User.username == username.strip().casefold())
        )

    def get_active_user(self, user_id: str) -> User | None:
        return self.session.scalar(
            select(User).where(User.id == user_id, User.is_active.is_(True))
        )

    def create_document(
        self,
        owner_id: str,
        name: str,
        *,
        is_public: bool = False,
    ) -> Document:
        cleaned = name.strip()
        if not cleaned or len(cleaned) > 255:
            raise ValueError("Document name must contain between 1 and 255 characters")
        document = Document(
            owner_id=owner_id,
            name=cleaned,
            is_public=is_public,
        )
        self.session.add(document)
        self.session.flush()
        return document

    def create_version(
        self,
        document_id: str,
        *,
        content_sha256: str,
        filename: str,
        storage_path: str,
        index_schema_version: str = "v3",
    ) -> DocumentVersion:
        version = DocumentVersion(
            document_id=document_id,
            content_sha256=content_sha256,
            filename=filename,
            storage_path=storage_path,
            index_schema_version=index_schema_version,
        )
        self.session.add(version)
        self.session.flush()
        return version

    def activate_version(self, document_id: str, version_id: str) -> None:
        document = self.session.get(Document, document_id)
        version = self.session.get(DocumentVersion, version_id)
        if (
            document is None
            or version is None
            or version.document_id != document_id
            or document.deleted_at is not None
        ):
            raise ValueError("Version does not belong to the active document")
        if document.active_version_id and document.active_version_id != version_id:
            previous = self.session.get(DocumentVersion, document.active_version_id)
            if previous is not None:
                previous.status = "superseded"
        version.status = "active"
        version.activated_at = utc_now()
        document.active_version_id = version.id
        document.status = "ready"
        document.updated_at = utc_now()
        self.session.flush()

    @staticmethod
    def _access_predicate(user_id: str):
        return or_(
            Document.owner_id == user_id,
            Document.is_public.is_(True),
            exists().where(
                DocumentPermission.document_id == Document.id,
                DocumentPermission.user_id == user_id,
                DocumentPermission.role == "reader",
            ),
        )

    def list_accessible_documents(self, user_id: str) -> list[Document]:
        return list(
            self.session.scalars(
                select(Document)
                .where(
                    Document.deleted_at.is_(None),
                    self._access_predicate(user_id),
                )
                .order_by(Document.name, Document.id)
            )
        )

    def require_document(
        self,
        user_id: str,
        document_id: str,
        *,
        owner: bool = False,
    ) -> Document:
        conditions = [
            Document.id == document_id,
            Document.deleted_at.is_(None),
        ]
        conditions.append(
            Document.owner_id == user_id
            if owner
            else self._access_predicate(user_id)
        )
        document = self.session.scalar(select(Document).where(*conditions))
        if document is None:
            raise DocumentAccessError("Document not found")
        return document

    def allowed_active_version_ids(
        self,
        user_id: str,
        document_ids: list[str] | None = None,
    ) -> list[str]:
        query = select(Document.active_version_id).where(
            Document.deleted_at.is_(None),
            Document.status == "ready",
            Document.active_version_id.is_not(None),
            self._access_predicate(user_id),
        )
        if document_ids is not None:
            query = query.where(Document.id.in_(document_ids))
        return sorted(str(value) for value in self.session.scalars(query) if value)

    def get_version(self, version_id: str | None) -> DocumentVersion | None:
        return self.session.get(DocumentVersion, version_id) if version_id else None

    def active_version_for_document(
        self,
        user_id: str,
        document_id: str,
    ) -> DocumentVersion:
        document = self.require_document(user_id, document_id)
        version = self.get_version(document.active_version_id)
        if version is None or version.status != "active":
            raise DocumentAccessError("Document not found")
        return version

    def mark_version_indexed(
        self,
        version_id: str,
        *,
        page_count: int,
        chunk_count: int,
    ) -> None:
        version = self.session.get(DocumentVersion, version_id)
        if version is None:
            raise ValueError("Version not found")
        version.page_count = page_count
        version.chunk_count = chunk_count
        self.session.flush()

    def mark_version_failed(self, version_id: str) -> None:
        version = self.session.get(DocumentVersion, version_id)
        if version is None:
            return
        version.status = "failed"
        document = self.session.get(Document, version.document_id)
        if document is not None and document.active_version_id is None:
            document.status = "failed"
        self.session.flush()

    def grant_reader(
        self,
        owner_id: str,
        document_id: str,
        reader_id: str,
    ) -> DocumentPermission:
        document = self.require_document(owner_id, document_id, owner=True)
        reader = self.get_active_user(reader_id)
        if reader is None or reader.id == document.owner_id:
            raise ValueError("Reader is not eligible for this grant")
        existing = self.session.scalar(
            select(DocumentPermission).where(
                DocumentPermission.document_id == document_id,
                DocumentPermission.user_id == reader_id,
            )
        )
        if existing is not None:
            return existing
        permission = DocumentPermission(
            document_id=document_id,
            user_id=reader_id,
            role="reader",
        )
        self.session.add(permission)
        self.session.flush()
        return permission

    def revoke_reader(self, owner_id: str, document_id: str, reader_id: str) -> None:
        self.require_document(owner_id, document_id, owner=True)
        permission = self.session.scalar(
            select(DocumentPermission).where(
                DocumentPermission.document_id == document_id,
                DocumentPermission.user_id == reader_id,
            )
        )
        if permission is not None:
            self.session.delete(permission)
            self.session.flush()
