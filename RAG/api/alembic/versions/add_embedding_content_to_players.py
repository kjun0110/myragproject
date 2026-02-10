"""Add embedding_content to players (temporary, for debugging)

Revision ID: add_embedding_content
Revises: add_embedding_to_players
Create Date: 2026-02-10 00:00:00.000000

임시: 어떤 텍스트로 임베딩했는지 확인용 컬럼.
"""

from typing import Sequence, Union

from alembic import op


revision: str = "add_embedding_content"
down_revision: Union[str, None] = "add_embedding_to_players"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE players
        ADD COLUMN IF NOT EXISTS embedding_content TEXT;
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE players DROP COLUMN IF EXISTS embedding_content;")
