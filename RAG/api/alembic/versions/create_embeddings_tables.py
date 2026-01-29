"""Create players_embeddings, teams_embeddings, stadiums_embeddings and schedules_embeddings tables

Revision ID: a1b2c3d4e5f6
Revises: dcd06a6b8416
Create Date: 2026-01-28 01:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'dcd06a6b8416'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create players_embeddings table
    op.create_table(
        'players_embeddings',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('player_id', sa.BigInteger(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(768), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['player_id'], ['players.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create HNSW index on players_embeddings.embedding
    op.execute(
        """
        CREATE INDEX idx_players_embeddings ON players_embeddings 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64);
        """
    )
    
    # Create teams_embeddings table
    op.create_table(
        'teams_embeddings',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('team_id', sa.BigInteger(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(768), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create HNSW index on teams_embeddings.embedding
    op.execute(
        """
        CREATE INDEX idx_teams_embeddings ON teams_embeddings 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64);
        """
    )
    
    # Create stadiums_embeddings table
    op.create_table(
        'stadiums_embeddings',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('stadium_id', sa.BigInteger(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(768), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['stadium_id'], ['stadiums.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create HNSW index on stadiums_embeddings.embedding
    op.execute(
        """
        CREATE INDEX idx_stadiums_embeddings ON stadiums_embeddings 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64);
        """
    )
    
    # Create schedules_embeddings table
    op.create_table(
        'schedules_embeddings',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('schedule_id', sa.BigInteger(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(768), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['schedule_id'], ['schedules.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create HNSW index on schedules_embeddings.embedding
    op.execute(
        """
        CREATE INDEX idx_schedules_embeddings ON schedules_embeddings 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64);
        """
    )


def downgrade() -> None:
    # Drop indexes first
    op.execute('DROP INDEX IF EXISTS idx_schedules_embeddings;')
    op.execute('DROP INDEX IF EXISTS idx_stadiums_embeddings;')
    op.execute('DROP INDEX IF EXISTS idx_teams_embeddings;')
    op.execute('DROP INDEX IF EXISTS idx_players_embeddings;')
    
    # Drop tables
    op.drop_table('schedules_embeddings')
    op.drop_table('stadiums_embeddings')
    op.drop_table('teams_embeddings')
    op.drop_table('players_embeddings')
