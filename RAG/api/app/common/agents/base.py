"""LLM 모델 기본 인터페이스."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings


class BaseLLM(ABC):
    """LLM 모델 기본 인터페이스."""

    @abstractmethod
    def get_langchain_model(self) -> BaseChatModel:
        """LangChain 호환 모델 반환."""
        pass

    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """프롬프트 실행 및 응답 반환."""
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        """스트리밍 응답 생성."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환."""
        pass


class BaseEmbedding(ABC):
    """Embedding 모델 기본 인터페이스."""

    @abstractmethod
    def get_langchain_embeddings(self) -> Embeddings:
        """LangChain 호환 Embeddings 반환."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 벡터로 변환."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환."""
        pass
