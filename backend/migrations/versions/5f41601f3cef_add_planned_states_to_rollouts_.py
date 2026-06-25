"""add planned_states to rollouts + predicted_user_state to data_fetches

Revision ID: 5f41601f3cef
Revises: 56d0c4d72d81
Create Date: 2026-05-23 15:10:07.869773

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5f41601f3cef'
down_revision: Union[str, Sequence[str], None] = '56d0c4d72d81'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Note: server_default supplies '[]' for any pre-existing rollouts so the NOT NULL
    # constraint can be satisfied during the column add. New inserts use the Python default.
    with op.batch_alter_table('rollouts', schema=None) as batch_op:
        batch_op.add_column(sa.Column(
            'planned_states', sa.JSON(), nullable=False, server_default=sa.text("'[]'"),
        ))
    with op.batch_alter_table('data_fetches', schema=None) as batch_op:
        batch_op.add_column(sa.Column('predicted_user_state', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('data_fetches', schema=None) as batch_op:
        batch_op.drop_column('predicted_user_state')
    with op.batch_alter_table('rollouts', schema=None) as batch_op:
        batch_op.drop_column('planned_states')
