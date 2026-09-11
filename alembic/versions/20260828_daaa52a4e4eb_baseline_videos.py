"""baseline videos

Revision ID: daaa52a4e4eb
Revises: 
Create Date: 2026-08-28 19:06:38.330138

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'daaa52a4e4eb'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Reproduce the `videos` table exactly as it already exists in production (created by
    the pre-alembic `init_db()`'s `create_all`). This migration is never actually run against
    that DB - it's `alembic stamp`'d there instead - but it lets a fresh DB reach the same
    state via `alembic upgrade head`."""
    op.create_table(
        "videos",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("filepath", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("format", sa.String(), nullable=False),
        sa.Column("duration_sec", sa.Float(), nullable=True),
        sa.Column("resolution", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="uploaded"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("videos")
