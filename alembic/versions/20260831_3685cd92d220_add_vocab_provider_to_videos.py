"""add vocab_provider to videos

Revision ID: 3685cd92d220
Revises: 710e7ce99a9a
Create Date: 2026-08-31 13:14:39.281434

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3685cd92d220'
down_revision: Union[str, Sequence[str], None] = '710e7ce99a9a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('videos', sa.Column('vocab_provider', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('videos', 'vocab_provider')
