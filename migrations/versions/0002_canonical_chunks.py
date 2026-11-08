"""Add canonical chunks and replacement ancestry."""

import sqlalchemy as sa
from alembic import op

revision = "0002_canonical_chunks"
down_revision = "0001_identity_catalog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_versions",
        sa.Column("based_on_version_id", sa.String(36), nullable=True),
    )
    op.add_column(
        "document_versions",
        sa.Column("derived_cleaned_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("version_id", sa.String(36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(32), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["version_id"], ["document_versions.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("version_id", "ordinal", name="uq_chunk_version_ordinal"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_version_id", "chunks", ["version_id"])
    op.create_index(
        "ix_chunks_document_version", "chunks", ["document_id", "version_id"]
    )


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_column("document_versions", "derived_cleaned_at")
    op.drop_column("document_versions", "based_on_version_id")
