"""Add embedding column to players and drop players_embeddings

Revision ID: add_embedding_to_players
Revises: f1e2d3c4b5a6
Create Date: 2026-02-10 00:00:00.000000

HNSW 통합 전략:
- players 테이블에 embedding 컬럼 추가 (Vector(768), nullable=True)
- HNSW 인덱스 생성 (players.embedding)
- players_embeddings 테이블 삭제

마이그레이션 순서:
1. players 테이블에 embedding 컬럼 추가
2. (선택) 기존 players_embeddings 데이터를 players로 복사
3. HNSW 인덱스 생성
4. players_embeddings 테이블 삭제
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "add_embedding_to_players"
down_revision: Union[str, None] = "f1e2d3c4b5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    HNSW 통합 마이그레이션 실행.
    
    주의: 기존 players_embeddings 데이터는 복사하지 않습니다.
    데이터 업로드 시 임베딩 배치 작업으로 새로 생성됩니다.
    """
    # 1. pgvector extension 확인 (이미 있을 것)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 2. players 테이블에 embedding 컬럼 추가
    op.execute("""
        ALTER TABLE players 
        ADD COLUMN IF NOT EXISTS embedding vector(768);
    """)
    
    # 3. HNSW 인덱스 생성 (players.embedding)
    # 데이터 규모에 따라 m, ef_construction 조정 가능
    op.execute("""
        CREATE INDEX IF NOT EXISTS players_embedding_hnsw_idx 
        ON players 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    
    # 4. players_embeddings 테이블 삭제
    # 주의: 기존 데이터는 백업 권장
    op.execute("DROP TABLE IF EXISTS players_embeddings CASCADE;")


def downgrade() -> None:
    """
    롤백: players.embedding 제거, players_embeddings 재생성.
    
    주의: 데이터 손실 가능성 있음. 프로덕션에서는 신중히 사용.
    """
    # 1. players_embeddings 테이블 재생성
    op.execute("""
        CREATE TABLE IF NOT EXISTS players_embeddings (
            id BIGSERIAL PRIMARY KEY,
            player_id BIGINT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            embedding vector(768) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    
    # 2. players_embeddings HNSW 인덱스 재생성
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_players_embeddings
        ON players_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    
    # 3. players 테이블의 인덱스 삭제
    op.execute("DROP INDEX IF EXISTS players_embedding_hnsw_idx;")
    
    # 4. players 테이블의 embedding 컬럼 삭제
    op.execute("ALTER TABLE players DROP COLUMN IF EXISTS embedding;")
