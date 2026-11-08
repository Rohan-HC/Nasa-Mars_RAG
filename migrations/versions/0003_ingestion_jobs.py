"""Add durable ingestion jobs with lease and retry state."""

import sqlalchemy as sa
from alembic import op

revision = "0003_ingestion_jobs"
down_revision = "0002_canonical_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("owner_id", sa.String(36), nullable=False),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("version_id", sa.String(36), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("upload_path", sa.String(1024), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("lease_owner", sa.String(128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(), nullable=True),
        sa.Column("error_class", sa.String(64), nullable=True),
        sa.Column("result_chunks", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["version_id"], ["document_versions.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("version_id", name="uq_ingestion_job_version"),
    )
    op.create_index("ix_ingestion_jobs_owner_id", "ingestion_jobs", ["owner_id"])
    op.create_index(
        "ix_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"]
    )
    op.create_index(
        "ix_ingestion_jobs_version_id", "ingestion_jobs", ["version_id"]
    )
    op.create_index(
        "ix_ingestion_jobs_claim",
        "ingestion_jobs",
        ["status", "next_attempt_at", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("ingestion_jobs")
