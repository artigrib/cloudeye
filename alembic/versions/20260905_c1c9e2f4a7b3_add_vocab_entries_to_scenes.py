"""add vocab_entries to scenes

Revision ID: c1c9e2f4a7b3
Revises: 9af4bc4316db
Create Date: 2026-09-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c1c9e2f4a7b3'
down_revision: Union[str, Sequence[str], None] = '9af4bc4316db'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('scenes', sa.Column('vocab_entries', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('scenes', 'vocab_entries')
