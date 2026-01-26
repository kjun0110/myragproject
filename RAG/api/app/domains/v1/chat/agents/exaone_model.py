"""Exaone 모델 구현체."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from langchain_core.language_models import BaseChatModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
agents_dir = current_file.parent  # api/app/domains/chat/agents/
chat_dir = agents_dir.parent  # api/app/domains/chat/
domains_dir = chat_dir.parent  # api/app/domains/
app_dir = domains_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

# BaseLLM 인터페이스 import
from app.common.agents.base import BaseLLM


class ExaoneLLM(BaseLLM):
    """Exaone LLM 모델 구현체."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_id: str = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        device_map: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        """Exaone 모델 초기화.

        Args:
            model_path: 로컬 모델 경로 (None이면 model_id 사용)
            model_id: HuggingFace 모델 ID
            device_map: 디바이스 매핑 ("auto", "cpu", "cuda" 등)
            dtype: 토치 데이터 타입 ("auto", "float16", "float32" 등)
            trust_remote_code: 원격 코드 신뢰 여부
        """
        self.model_path = model_path
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        # 모델 경로 결정 (HuggingFace 캐시 우선 사용)
        if model_path and Path(model_path).exists():
            self._load_path = model_path
        else:
            # HuggingFace 캐시 사용 (model_id 사용)
            self._load_path = model_id

        self.model = None
        self.tokenizer = None
        self._langchain_model = None

    def _load_model(self):
        """모델 로드 (lazy loading)."""
        if self.model is not None:
            return

        print(f"[INFO] Exaone 모델 로드 중...")

        # 공통 모델 로더 사용 (HuggingFace 캐시 활용)
        from app.common.loaders import ModelLoader

        # 양자화 여부 결정
        use_quantization = self.dtype in ["auto", "float16", "bfloat16"]

        # 아답터 없이 베이스 모델만 로드
        self.model, self.tokenizer = ModelLoader.load_exaone_model(
            adapter_name=None,
            use_quantization=use_quantization,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )

        print("[OK] Exaone 모델 로드 완료")

    def get_langchain_model(self) -> BaseChatModel:
        """LangChain 호환 모델 반환."""
        if self._langchain_model is None:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
            from transformers import pipeline

            self._load_model()

            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

            # langchain_huggingface의 HuggingFacePipeline 사용 (ChatHuggingFace와 호환)
            hf_pipeline = HuggingFacePipeline(pipeline=pipe)
            self._langchain_model = ChatHuggingFace(llm=hf_pipeline)

        return self._langchain_model

    def invoke(self, prompt: str, **kwargs) -> str:
        """프롬프트 실행 및 응답 반환."""
        self._load_model()

        max_new_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 프롬프트 부분 제거
        if response.startswith(prompt):
            response = response[len(prompt) :].strip()

        return response

    def stream(self, prompt: str, **kwargs):
        """스트리밍 응답 생성."""
        # 간단한 구현: 전체 응답을 한 번에 반환
        response = self.invoke(prompt, **kwargs)
        yield response

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환."""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "load_path": self._load_path,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }
