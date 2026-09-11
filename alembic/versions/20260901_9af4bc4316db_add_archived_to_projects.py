"""add archived to projects

Revision ID: 9af4bc4316db
Revises: 52dd0a2d3ffa
Create Date: 2026-09-01 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9af4bc4316db'
down_revision: Union[str, Sequence[str], None] = '52dd0a2d3ffa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # server_default backfills existing rows to false - projects is non-empty, and a
    # NOT NULL column added via ALTER TABLE needs one for that to succeed.
    op.add_column(
        'projects',
        sa.Column('archived', sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('projects', 'archived')
