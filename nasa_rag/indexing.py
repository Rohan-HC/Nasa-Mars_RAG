from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import pymupdf
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from nasa_rag.catalog import CatalogRepository, DocumentAccessError
from nasa_rag.db import Database
from nasa_rag.models import Chunk
from nasa_rag.retrieval import IndexedChunk


INDEX_SCHEMA_VERSION = "v3"


class ConcurrentReplacementError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractedChunk:
    page: int
    section: str
    text: str


@dataclass(frozen=True)
class IndexingPlan:
    document_id: str
    version_id: str
    action: Literal["index", "skip"]


@dataclass(frozen=True)
class IndexingResult:
    document_id: str
    version_id: str
    status: Literal["indexed", "already_indexed"]
    chunks: int
    pages: int
    cleanup_complete: bool = True


class ChunkExtractor(Protocol):
    def extract(self, path: Path) -> list[ExtractedChunk]: ...


class DenseIndex(Protocol):
    def upsert(self, chunks: list[IndexedChunk]) -> None: ...

    def verify(self, chunks: list[IndexedChunk]) -> bool: ...

    def version_chunk_ids(self, version_id: str) -> set[str]: ...

    def delete_version(self, version_id: str) -> None: ...


class PdfChunkExtractor:
    _references_heading = re.compile(
        r"(?im)^[ \t]*(references|bibliography)[ \t]*:?[ \t]*$"
    )

    def __init__(self, *, chunk_size: int = 800, chunk_overlap: int = 120):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def extract(self, path: Path) -> list[ExtractedChunk]:
        pages: list[LangChainDocument] = []
        in_references = False
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text", sort=False)
                if not text.strip():
                    continue
                if in_references:
                    sections = [("references", text)]
                else:
                    match = self._references_heading.search(text)
                    if match:
                        sections = [
                            ("content", text[: match.start()]),
                            ("references", text[match.end() :]),
                        ]
                        in_references = True
                    else:
                        sections = [("content", text)]
                for section, section_text in sections:
                    if section_text.strip():
                        pages.append(
                            LangChainDocument(
                                page_content=section_text,
                                metadata={"page": page_number, "section": section},
                            )
                        )
        chunks = self.splitter.split_documents(pages)
        if not chunks:
            raise ValueError("No extractable text found")
        return [
            ExtractedChunk(
                page=int(chunk.metadata["page"]),
                section=str(chunk.metadata["section"]),
                text=chunk.page_content,
            )
            for chunk in chunks
        ]


class IndexingService:
    def __init__(
        self,
        database: Database,
        extractor: ChunkExtractor,
        dense_index: DenseIndex,
        *,
        index_schema_version: str = INDEX_SCHEMA_VERSION,
        storage_root: Path | None = None,
    ):
        self.database = database
        self.extractor = extractor
        self.dense_index = dense_index
        self.index_schema_version = index_schema_version
        self.storage_root = storage_root.resolve() if storage_root else None

    def prepare_new_document(
        self,
        *,
        owner_id: str,
        name: str,
        storage_path: Path,
        content_sha256: str,
    ) -> IndexingPlan:
        with self.database.session() as session:
            catalog = CatalogRepository(session)
            document = catalog.create_document(owner_id, name)
            version = catalog.create_version(
                document.id,
                content_sha256=content_sha256,
                filename=name,
                storage_path=str(storage_path),
                index_schema_version=self.index_schema_version,
                based_on_version_id=None,
            )
            return IndexingPlan(document.id, version.id, "index")

    def prepare_replacement(
        self,
        *,
        owner_id: str,
        document_id: str,
        storage_path: Path,
        content_sha256: str,
        check_integrity: bool = True,
    ) -> IndexingPlan:
        with self.database.session() as session:
            catalog = CatalogRepository(session)
            document = catalog.require_document(owner_id, document_id, owner=True)
            active_id = document.active_version_id
            active = catalog.get_version(active_id)
            if (
                active is not None
                and active.content_sha256 == content_sha256
                and active.index_schema_version == self.index_schema_version
            ):
                if check_integrity and self._integrity_established(
                    catalog, active.id, active.chunk_count
                ):
                    return IndexingPlan(document.id, active.id, "skip")
                return IndexingPlan(document.id, active.id, "index")

            version = catalog.find_version(
                document.id,
                content_sha256,
                self.index_schema_version,
            )
            if version is None:
                version = catalog.create_version(
                    document.id,
                    content_sha256=content_sha256,
                    filename=document.name,
                    storage_path=str(storage_path),
                    index_schema_version=self.index_schema_version,
                    based_on_version_id=active_id,
                )
            else:
                version.filename = document.name
                version.storage_path = str(storage_path)
                version.based_on_version_id = active_id
                version.derived_cleaned_at = None
                if version.id != active_id:
                    version.status = "staged"
                session.flush()
            return IndexingPlan(document.id, version.id, "index")

    def process(self, version_id: str) -> IndexingResult:
        with self.database.session() as session:
            catalog = CatalogRepository(session)
            version = catalog.get_version(version_id)
            if version is None:
                raise ValueError("Version not found")
            document = catalog.require_document(
                self._document_owner(session, version.document_id),
                version.document_id,
                owner=True,
            )
            if version.status == "active" and self._integrity_established(
                catalog, version.id, version.chunk_count
            ):
                return IndexingResult(
                    document.id,
                    version.id,
                    "already_indexed",
                    version.chunk_count or 0,
                    version.page_count or 0,
                )
            storage_path = Path(version.storage_path)
            filename = version.filename
            document_id = version.document_id
            expected_active_id = version.based_on_version_id
            version_was_active = version.status == "active"

        extracted = self.extractor.extract(storage_path)
        chunks = self._indexed_chunks(
            document_id=document_id,
            version_id=version_id,
            filename=filename,
            extracted=extracted,
        )
        previous_id: str | None = None
        try:
            self.dense_index.upsert(chunks)
            if not self.dense_index.verify(chunks):
                raise RuntimeError("Dense index verification failed")
            records = [
                Chunk(
                    id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    version_id=chunk.version_id,
                    ordinal=ordinal,
                    page=chunk.page,
                    section=chunk.section,
                    filename=chunk.filename,
                    text=chunk.text,
                )
                for ordinal, chunk in enumerate(chunks, start=1)
            ]
            with self.database.session() as session:
                catalog = CatalogRepository(session)
                catalog.replace_chunks(version_id, records)
                version = catalog.get_version(version_id)
                if version is None:
                    raise ValueError("Version not found")
                version.page_count = len({chunk.page for chunk in chunks})
                version.chunk_count = len(chunks)
                try:
                    previous_id = catalog.promote_version(
                        document_id,
                        version_id,
                        expected_active_version_id=expected_active_id,
                    )
                except RuntimeError as error:
                    raise ConcurrentReplacementError(
                        "A newer document version is already active"
                    ) from error
        except Exception:
            self._discard_failed_version(version_id, version_was_active)
            raise

        cleanup_complete = True
        if previous_id and previous_id != version_id:
            cleanup_complete = self._cleanup_version(previous_id)
        return IndexingResult(
            document_id,
            version_id,
            "indexed",
            len(chunks),
            len({chunk.page for chunk in chunks}),
            cleanup_complete,
        )

    def delete_document(self, owner_id: str, document_id: str) -> bool:
        with self.database.session() as session:
            catalog = CatalogRepository(session)
            document = catalog.require_document(owner_id, document_id, owner=True)
            versions = catalog.all_versions_for_document(document.id)
            version_ids = catalog.soft_delete_document(owner_id, document_id)
            storage_paths = [Path(version.storage_path) for version in versions]
        cleanup_results = [
            self._cleanup_version(version_id) for version_id in version_ids
        ]
        for storage_path in storage_paths:
            self._remove_managed_source(storage_path)
        return all(cleanup_results)

    def reconcile_derived_data(self) -> int:
        with self.database.session() as session:
            version_ids = [
                version.id for version in CatalogRepository(session).stale_derived_versions()
            ]
        return sum(1 for version_id in version_ids if self._cleanup_version(version_id))

    def _integrity_established(
        self,
        catalog: CatalogRepository,
        version_id: str,
        expected_count: int | None,
    ) -> bool:
        if not expected_count or catalog.chunk_count(version_id) != expected_count:
            return False
        database_ids = catalog.chunk_ids(version_id)
        return (
            len(database_ids) == expected_count
            and self.dense_index.version_chunk_ids(version_id) == database_ids
        )

    def _indexed_chunks(
        self,
        *,
        document_id: str,
        version_id: str,
        filename: str,
        extracted: list[ExtractedChunk],
    ) -> list[IndexedChunk]:
        return [
            IndexedChunk(
                chunk_id=(
                    f"{document_id}:{version_id}:{self.index_schema_version}:{ordinal}"
                ),
                document_id=document_id,
                version_id=version_id,
                filename=filename,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
            )
            for ordinal, chunk in enumerate(extracted, start=1)
        ]

    def _discard_failed_version(self, version_id: str, was_active: bool) -> None:
        try:
            self.dense_index.delete_version(version_id)
        except Exception:
            pass
        with self.database.session() as session:
            catalog = CatalogRepository(session)
            catalog.delete_chunks(version_id)
            if not was_active:
                catalog.mark_version_failed(version_id)

    def _cleanup_version(self, version_id: str) -> bool:
        try:
            self.dense_index.delete_version(version_id)
        except Exception:
            return False
        with self.database.session() as session:
            catalog = CatalogRepository(session)
            catalog.delete_chunks(version_id)
            catalog.mark_derived_cleaned(version_id)
        return True

    def _remove_managed_source(self, path: Path) -> bool:
        if self.storage_root is None:
            return False
        try:
            resolved = path.resolve()
            resolved.relative_to(self.storage_root)
        except (OSError, ValueError):
            return False
        try:
            resolved.unlink(missing_ok=True)
        except OSError:
            return False
        return True

    @staticmethod
    def _document_owner(session, document_id: str) -> str:
        from nasa_rag.models import Document

        document = session.get(Document, document_id)
        if document is None or document.deleted_at is not None:
            raise DocumentAccessError("Document not found")
        return document.owner_id
