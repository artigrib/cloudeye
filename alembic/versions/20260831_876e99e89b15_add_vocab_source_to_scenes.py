"""add vocab_source to scenes

Revision ID: 876e99e89b15
Revises: a1b17077b84f
Create Date: 2026-08-31 07:22:06.373589

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '876e99e89b15'
down_revision: Union[str, Sequence[str], None] = 'a1b17077b84f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('scenes', sa.Column('vocab_source', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('scenes', 'vocab_source')
