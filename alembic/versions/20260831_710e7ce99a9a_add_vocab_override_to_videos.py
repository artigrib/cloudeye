"""add vocab_override to videos

Revision ID: 710e7ce99a9a
Revises: 876e99e89b15
Create Date: 2026-08-31 09:30:08.441435

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '710e7ce99a9a'
down_revision: Union[str, Sequence[str], None] = '876e99e89b15'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('videos', sa.Column('vocab_override', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('videos', 'vocab_override')
