"""Add incremental FTS5 BM25 index over canonical chunks."""

from alembic import op


revision = "0004_fts5_bm25"
down_revision = "0003_ingestion_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            chunk_id UNINDEXED,
            document_id UNINDEXED,
            version_id UNINDEXED,
            filename UNINDEXED,
            page UNINDEXED,
            section UNINDEXED,
            text,
            tokenize='unicode61'
        )
        """
    )
    op.execute(
        """
        INSERT INTO chunk_fts(
            chunk_id, document_id, version_id, filename, page, section, text
        )
        SELECT id, document_id, version_id, filename, page, section, text
        FROM chunks
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE chunk_fts")
