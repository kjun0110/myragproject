"""LLM 모델 팩토리 - 모델 생성 및 주입."""

from typing import Optional, Dict, Any, List
from .base import BaseLLM, BaseEmbedding


class LLMFactory:
    """LLM 모델 팩토리."""

    _models: Dict[str, BaseLLM] = {}
    _default_model: Optional[str] = None

    @classmethod
    def register(cls, name: str, model: BaseLLM, is_default: bool = False) -> None:
        """모델 등록."""
        cls._models[name] = model
        if is_default or cls._default_model is None:
            cls._default_model = name

    @classmethod
    def get(cls, name: Optional[str] = None) -> BaseLLM:
        """모델 가져오기."""
        model_name = name or cls._default_model
        if model_name is None:
            raise ValueError("모델이 등록되지 않았습니다.")
        if model_name not in cls._models:
            raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다.")
        return cls._models[model_name]

    @classmethod
    def list_models(cls) -> List[str]:
        """등록된 모델 목록 반환."""
        return list(cls._models.keys())

    @classmethod
    def get_default(cls) -> Optional[str]:
        """기본 모델 이름 반환."""
        return cls._default_model


class EmbeddingFactory:
    """Embedding 모델 팩토리."""

    _embeddings: Dict[str, BaseEmbedding] = {}
    _default_embedding: Optional[str] = None

    @classmethod
    def register(
        cls, name: str, embedding: BaseEmbedding, is_default: bool = False
    ) -> None:
        """Embedding 모델 등록."""
        cls._embeddings[name] = embedding
        if is_default or cls._default_embedding is None:
            cls._default_embedding = name

    @classmethod
    def get(cls, name: Optional[str] = None) -> BaseEmbedding:
        """Embedding 모델 가져오기."""
        embedding_name = name or cls._default_embedding
        if embedding_name is None:
            raise ValueError("Embedding 모델이 등록되지 않았습니다.")
        if embedding_name not in cls._embeddings:
            raise ValueError(f"Embedding 모델 '{embedding_name}'을 찾을 수 없습니다.")
        return cls._embeddings[embedding_name]

    @classmethod
    def list_embeddings(cls) -> List[str]:
        """등록된 Embedding 모델 목록 반환."""
        return list(cls._embeddings.keys())

    @classmethod
    def get_default(cls) -> Optional[str]:
        """기본 Embedding 모델 이름 반환."""
        return cls._default_embedding
