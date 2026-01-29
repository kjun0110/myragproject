"""Recreate embeddings tables if missing

Revision ID: f1e2d3c4b5a6
Revises: a1b2c3d4e5f6
Create Date: 2026-01-29 00:00:00.000000

의도:
- 운영 DB에서 임베딩 테이블이 실수로 DROP 된 경우에도
  Alembic `upgrade head`로 안전하게 복구할 수 있도록 "IF NOT EXISTS" 기반 복구 마이그레이션을 제공합니다.
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "f1e2d3c4b5a6"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # pgvector extension (safe)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # players_embeddings
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS players_embeddings (
          id BIGSERIAL PRIMARY KEY,
          player_id BIGINT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
          content TEXT NOT NULL,
          embedding vector(768) NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_players_embeddings
        ON players_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    # teams_embeddings
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS teams_embeddings (
          id BIGSERIAL PRIMARY KEY,
          team_id BIGINT NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
          content TEXT NOT NULL,
          embedding vector(768) NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_teams_embeddings
        ON teams_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    # stadiums_embeddings
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS stadiums_embeddings (
          id BIGSERIAL PRIMARY KEY,
          stadium_id BIGINT NOT NULL REFERENCES stadiums(id) ON DELETE CASCADE,
          content TEXT NOT NULL,
          embedding vector(768) NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stadiums_embeddings
        ON stadiums_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )

    # schedules_embeddings
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS schedules_embeddings (
          id BIGSERIAL PRIMARY KEY,
          schedule_id BIGINT NOT NULL REFERENCES schedules(id) ON DELETE CASCADE,
          content TEXT NOT NULL,
          embedding vector(768) NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_schedules_embeddings
        ON schedules_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )


def downgrade() -> None:
    # Drop indexes first (safe)
    op.execute("DROP INDEX IF EXISTS idx_schedules_embeddings;")
    op.execute("DROP INDEX IF EXISTS idx_stadiums_embeddings;")
    op.execute("DROP INDEX IF EXISTS idx_teams_embeddings;")
    op.execute("DROP INDEX IF EXISTS idx_players_embeddings;")

    # Drop tables (safe)
    op.execute("DROP TABLE IF EXISTS schedules_embeddings;")
    op.execute("DROP TABLE IF EXISTS stadiums_embeddings;")
    op.execute("DROP TABLE IF EXISTS teams_embeddings;")
    op.execute("DROP TABLE IF EXISTS players_embeddings;")

