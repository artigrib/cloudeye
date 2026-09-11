"""add vocab_model to scenes

Revision ID: 6c4120308ca8
Revises: 3685cd92d220
Create Date: 2026-08-31 13:14:40.040257

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6c4120308ca8'
down_revision: Union[str, Sequence[str], None] = '3685cd92d220'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('scenes', sa.Column('vocab_model', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('scenes', 'vocab_model')
