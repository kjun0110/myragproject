#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""공통 모델 로더 유틸리티.

HuggingFace 캐시를 활용한 베이스 모델 로딩과
로컬 아답터 로딩을 분리하여 관리합니다.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# 프로젝트 루트 경로
current_file = Path(__file__).resolve()
loaders_dir = current_file.parent  # api/app/common/loaders/
common_dir = loaders_dir.parent  # api/app/common/
app_dir = common_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))


class ModelLoader:
    """모델 로더 기본 클래스."""

    # HuggingFace 모델 ID 상수
    EXAONE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    KOELECTRA_MODEL_ID = "monologg/koelectra-small-v3-discriminator"

    # 로컬 아답터 경로 (환경 변수로 오버라이드 가능)
    EXAONE_ADAPTER_DIR = os.getenv(
        "EXAONE_ADAPTER_DIR",
        str(api_dir / "artifacts" / "exaone" / "spam_adapter"),
    )
    KOELECTRA_ADAPTER_DIR = os.getenv(
        "KOELECTRA_ADAPTER_DIR",
        str(api_dir / "artifacts" / "koelectra" / "spam_adapter"),
    )

    # EXAONE 커스텀 코드 경로 (trust_remote_code 대신 로컬 경로 사용)
    EXAONE_LOCAL_MODEL_DIR = str(api_dir / "artifacts" / "exaone" / "exaone3.5-2.4b")

    @staticmethod
    def _get_latest_adapter_dir(adapter_base_dir: str, model_name: str) -> Optional[str]:
        """최신 아답터 디렉터리 경로를 반환합니다.

        Args:
            adapter_base_dir: 아답터 베이스 디렉터리
            model_name: 모델 이름 (예: "exaone3.5-2.4b-spam-lora")

        Returns:
            최신 아답터 디렉터리 경로 또는 None
        """
        adapter_dir = Path(adapter_base_dir) / model_name

        if not adapter_dir.exists():
            return None

        # 타임스탬프 기반 서브디렉터리 찾기 (예: 20260114-161237)
        subdirs = [d for d in adapter_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return None

        # 최신 디렉터리 반환 (수정 시간 기준)
        latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
        return str(latest_dir)

    @staticmethod
    def load_exaone_model(
        adapter_name: Optional[str] = None,
        use_quantization: bool = True,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> Tuple:
        """EXAONE 모델을 로드합니다.

        Args:
            adapter_name: 아답터 이름 (예: "exaone3.5-2.4b-spam-lora")
                         None이면 베이스 모델만 로드
            use_quantization: 4-bit 양자화 사용 여부
            device_map: 디바이스 매핑 ("auto", "cpu", "cuda" 등)
            trust_remote_code: 원격 코드 신뢰 여부 (커스텀 모델링 코드 사용)

        Returns:
            (model, tokenizer) 튜플
        """
        print("[INFO] EXAONE 모델 로딩 시작...")

        # EXAONE은 HuggingFace 캐시 사용 (가중치는 캐시에서)
        # 커스텀 코드는 로컬 경로에서 참조 (modeling_exaone.py, configuration_exaone.py)
        model_path = ModelLoader.EXAONE_MODEL_ID
        print(f"[INFO] HuggingFace 캐시 활용: {model_path}")
        
        # 커스텀 코드는 HuggingFace 캐시에서 자동으로 로드됨
        # trust_remote_code=True로 설정되어 있으면 캐시에 자동 저장됨

        # 토크나이저 로드
        print("[INFO] 토크나이저 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        # pad_token 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 양자화 설정
        quantization_config = None
        if use_quantization and torch.cuda.is_available():
            print("[INFO] 4-bit 양자화 설정 적용")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # 베이스 모델 로드 (최적화 옵션 적용)
        print("[INFO] 베이스 모델 로드 중...")
        
        # 최적화 옵션 (환경 변수로 제어 가능)
        use_safetensors = os.getenv("USE_SAFETENSORS", "true").lower() == "true"
        low_cpu_mem_usage = os.getenv("LOW_CPU_MEM_USAGE", "true").lower() == "true"
        
        # torch_dtype 설정 (양자화 사용 시에는 quantization_config가 처리)
        torch_dtype = None
        if not use_quantization and torch.cuda.is_available():
            torch_dtype = torch.bfloat16  # GPU에서 bfloat16 사용
        
        # max_memory 설정 (GPU 메모리 관리)
        max_memory = None
        if torch.cuda.is_available() and device_map == "auto":
            # GPU 메모리 제한 설정 (옵션)
            max_memory_env = os.getenv("MAX_MEMORY_GB")
            if max_memory_env:
                try:
                    max_memory_gb = float(max_memory_env)
                    max_memory = {0: f"{max_memory_gb}GB"}
                    print(f"[INFO] GPU 메모리 제한: {max_memory_gb}GB")
                except ValueError:
                    pass
        
        load_kwargs = {
            "quantization_config": quantization_config,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "use_safetensors": use_safetensors,
        }
        
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        
        if use_safetensors:
            print("[INFO] safetensors 형식 사용 시도 (빠른 로딩)")
        else:
            print("[WARNING] safetensors 비활성화 - PyTorch pickle 형식 사용 (느림)")
        if low_cpu_mem_usage:
            print("[INFO] 낮은 CPU 메모리 사용 모드 활성화")
        
        # 실제로 사용되는 파일 형식 확인을 위한 로깅
        import time
        load_start = time.time()
        
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            load_time = time.time() - load_start
            print(f"[INFO] 모델 로딩 완료 (소요 시간: {load_time:.2f}초)")
        except Exception as e:
            # safetensors가 없으면 자동으로 .bin 사용
            if "safetensors" in str(e).lower() or use_safetensors:
                print(f"[WARNING] safetensors 로딩 실패, .bin 형식으로 재시도: {e}")
                # safetensors 옵션 제거하고 재시도
                load_kwargs.pop("use_safetensors", None)
                load_start = time.time()
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **load_kwargs
                )
                load_time = time.time() - load_start
                print(f"[INFO] PyTorch pickle 형식으로 로딩 완료 (소요 시간: {load_time:.2f}초)")
            else:
                raise

        # 아답터 로드 (선택적)
        if adapter_name:
            adapter_path = ModelLoader._get_latest_adapter_dir(
                ModelLoader.EXAONE_ADAPTER_DIR, adapter_name
            )
            if adapter_path:
                print(f"[INFO] 아답터 로드 중: {adapter_path}")
                model = PeftModel.from_pretrained(base_model, adapter_path)
                print("[OK] 아답터 로드 완료")
            else:
                print(f"[WARNING] 아답터를 찾을 수 없음: {adapter_name}")
                print("[INFO] 베이스 모델만 사용합니다")
                model = base_model
        else:
            model = base_model

        model.eval()
        print("[OK] EXAONE 모델 로드 완료")
        return model, tokenizer

    @staticmethod
    def load_koelectra_model(
        adapter_name: Optional[str] = None,
        device: Optional[str] = None,
        num_labels: int = 2,
    ) -> Tuple:
        """KoELECTRA 모델을 로드합니다.

        Args:
            adapter_name: 아답터 이름 (예: "koelectra-small-v3-discriminator-spam-lora")
                         None이면 베이스 모델만 로드
            device: 디바이스 ("cuda", "cpu" 등). None이면 자동 선택
            num_labels: 분류 레이블 수 (기본값: 2)

        Returns:
            (model, tokenizer) 튜플
        """
        print("[INFO] KoELECTRA 모델 로딩 시작...")

        # HuggingFace Hub에서 자동 다운로드 (캐시 활용)
        model_path = ModelLoader.KOELECTRA_MODEL_ID
        print(f"[INFO] HuggingFace 캐시 활용: {model_path}")

        # 디바이스 설정
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] 디바이스: {device}")

        # 토크나이저 로드
        print("[INFO] 토크나이저 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 베이스 모델 로드 (최적화 옵션 적용)
        print("[INFO] 베이스 모델 로드 중...")
        
        # 최적화 옵션
        use_safetensors = os.getenv("USE_SAFETENSORS", "true").lower() == "true"
        low_cpu_mem_usage = os.getenv("LOW_CPU_MEM_USAGE", "true").lower() == "true"
        
        load_kwargs = {
            "num_labels": num_labels,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "use_safetensors": use_safetensors,
        }
        
        if use_safetensors:
            print("[INFO] safetensors 형식 사용 (빠른 로딩)")
        if low_cpu_mem_usage:
            print("[INFO] 낮은 CPU 메모리 사용 모드 활성화")
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            **load_kwargs
        )
        base_model.to(device)

        # 아답터 로드 (선택적)
        if adapter_name:
            adapter_path = ModelLoader._get_latest_adapter_dir(
                ModelLoader.KOELECTRA_ADAPTER_DIR, adapter_name
            )
            if adapter_path:
                print(f"[INFO] 아답터 로드 중: {adapter_path}")
                model = PeftModel.from_pretrained(base_model, adapter_path)
                print("[OK] 아답터 로드 완료")
            else:
                print(f"[WARNING] 아답터를 찾을 수 없음: {adapter_name}")
                print("[INFO] 베이스 모델만 사용합니다")
                model = base_model
        else:
            model = base_model

        model.eval()
        print("[OK] KoELECTRA 모델 로드 완료")
        return model, tokenizer


# 편의 함수들
def load_exaone_with_spam_adapter(**kwargs) -> Tuple:
    """EXAONE + 스팸 분류 아답터를 로드합니다."""
    return ModelLoader.load_exaone_model(
        adapter_name="exaone3.5-2.4b-spam-lora",
        **kwargs,
    )


def load_koelectra_with_spam_adapter(**kwargs) -> Tuple:
    """KoELECTRA + 스팸 분류 아답터를 로드합니다."""
    return ModelLoader.load_koelectra_model(
        adapter_name="koelectra-small-v3-discriminator-spam-lora",
        **kwargs,
    )


if __name__ == "__main__":
    # 테스트 코드
    print("=" * 60)
    print("모델 로더 테스트")
    print("=" * 60)

    # KoELECTRA 테스트
    print("\n[TEST] KoELECTRA 모델 로드")
    try:
        model, tokenizer = ModelLoader.load_koelectra_model()
        print("[SUCCESS] KoELECTRA 로드 성공")
    except Exception as e:
        print(f"[FAILED] KoELECTRA 로드 실패: {e}")

    # EXAONE 테스트
    print("\n[TEST] EXAONE 모델 로드")
    try:
        model, tokenizer = ModelLoader.load_exaone_model()
        print("[SUCCESS] EXAONE 로드 성공")
    except Exception as e:
        print(f"[FAILED] EXAONE 로드 실패: {e}")
