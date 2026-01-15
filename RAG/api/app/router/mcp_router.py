"""
KoELECTRA 게이트웨이 라우터.

POST /api/mcp/gate
이메일 텍스트를 받아 KoELECTRA 게이트웨이로 스팸 확률을 계산합니다.
상태 관리 기능을 포함하여 게이트웨이 결과를 저장하고 관리합니다.

POST /api/mcp/spam-analyze
EXAONE 툴을 사용하여 전체 스팸 분석을 수행합니다.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
router_dir = current_file.parent  # api/app/router/
app_dir = router_dir.parent  # api/app/
api_dir = app_dir.parent  # api/

sys.path.insert(0, str(api_dir))

router = APIRouter(prefix="/api/mcp", tags=["mcp"])

# ==================== 상태 관리 (인메모리) ====================
# 실제 운영 환경에서는 Redis나 DB를 사용하는 것을 권장합니다
_gate_results_cache: dict[str, dict] = {}  # {request_id: gate_result}


# ==================== 모델 로드 (전역 캐싱) ====================
_koelectra_model = None
_koelectra_tokenizer = None
_koelectra_loading = False
_koelectra_error = None


def load_koelectra_gate():
    """KoELECTRA 게이트웨이 모델을 로드합니다."""
    global _koelectra_model, _koelectra_tokenizer, _koelectra_loading, _koelectra_error

    if _koelectra_model is not None and _koelectra_tokenizer is not None:
        return _koelectra_model, _koelectra_tokenizer

    if _koelectra_error is not None:
        raise RuntimeError(f"이전 KoELECTRA 모델 로드 실패: {_koelectra_error}")

    if _koelectra_loading:
        raise RuntimeError("KoELECTRA 모델이 현재 로딩 중입니다.")

    _koelectra_loading = True
    try:
        print("[INFO] KoELECTRA 게이트웨이 모델 로딩 시작...")

        # Base 모델 경로
        model_dir = app_dir / "model" / "koelectra-small-v3-discriminator"
        if model_dir.exists() and (model_dir / "config.json").exists():
            base_model_path = str(model_dir)
        else:
            base_model_path = "monologg/koelectra-small-v3-discriminator"

        # LoRA 어댑터 경로
        lora_dir = app_dir / "model" / "koelectra-small-v3-discriminator-spam-lora"
        lora_adapter_path = None
        if lora_dir.exists():
            subdirs = [d for d in lora_dir.iterdir() if d.is_dir()]
            if subdirs:
                lora_adapter_path = str(max(subdirs, key=lambda x: x.stat().st_mtime))
            else:
                lora_adapter_path = str(lora_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] 디바이스: {device}")

        # 토크나이저 로드
        _koelectra_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # Base 모델 로드
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
        base_model.to(device)

        # LoRA 어댑터 로드 (있는 경우)
        if lora_adapter_path:
            _koelectra_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        else:
            _koelectra_model = base_model

        _koelectra_model.eval()
        print("[OK] KoELECTRA 게이트웨이 모델 로드 완료")
        _koelectra_loading = False
        _koelectra_error = None
        return _koelectra_model, _koelectra_tokenizer

    except Exception as e:
        _koelectra_loading = False
        error_msg = f"KoELECTRA 모델 로드 실패: {str(e)}"
        _koelectra_error = error_msg
        print(f"[ERROR] {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise RuntimeError(error_msg) from e


# ==================== 게이트웨이 함수 ====================
def predict_spam_probability(
    email_text: str, max_length: int = 512, request_id: Optional[str] = None
) -> dict:
    """텍스트에 대한 스팸 확률을 계산합니다.

    Args:
        email_text: 분류할 텍스트 (이메일 제목, 본문 등)
        max_length: 최대 토큰 길이
        request_id: 요청 ID (상태 관리용, 선택사항)

    Returns:
        {
            "spam_prob": float,
            "ham_prob": float,
            "label": str,
            "confidence": str,
            "latency_ms": float,
            "request_id": str (있는 경우)
        }
    """
    start_time = time.time()

    try:
        model, tokenizer = load_koelectra_gate()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 토크나이징
        inputs = tokenizer(
            email_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        spam_prob = probabilities[0][1].item()
        ham_prob = probabilities[0][0].item()
        label = "spam" if spam_prob > 0.5 else "ham"

        # 신뢰도 결정
        if abs(spam_prob - 0.5) < 0.1:
            confidence = "low"
        elif abs(spam_prob - 0.5) < 0.3:
            confidence = "medium"
        else:
            confidence = "high"

        latency_ms = (time.time() - start_time) * 1000

        gate_result = {
            "spam_prob": spam_prob,
            "ham_prob": ham_prob,
            "label": label,
            "confidence": confidence,
            "latency_ms": round(latency_ms, 2),
        }

        # 상태 관리: request_id가 있으면 결과를 캐시에 저장
        if request_id:
            gate_result["request_id"] = request_id
            _gate_results_cache[request_id] = gate_result

        print(
            f"[GATE] 스팸 확률: {spam_prob:.4f}, 레이블: {label}, 신뢰도: {confidence}"
        )

        return gate_result

    except Exception as e:
        print(f"[ERROR] 게이트웨이 처리 실패: {e}")
        raise


def should_call_exaone(spam_prob: float, thresholds: Optional[dict] = None) -> bool:
    """EXAONE Reader를 호출해야 하는지 결정합니다.

    Args:
        spam_prob: 스팸 확률 (0.0 ~ 1.0)
        thresholds: 임계치 설정 (None이면 기본값 사용)

    Returns:
        True: EXAONE Reader 호출 필요
        False: EXAONE Reader 호출 불필요
    """
    if thresholds is None:
        thresholds = {
            "low": 0.2,
            "high": 0.85,
            "ambiguous_low": 0.35,
            "ambiguous_high": 0.8,
        }

    # 명확한 구간: EXAONE 호출 불필요
    if spam_prob <= thresholds["low"]:
        return False  # 즉시 deliver
    if spam_prob >= thresholds["high"]:
        return False  # 즉시 quarantine

    # 애매 구간: EXAONE 호출 필요
    if thresholds["ambiguous_low"] <= spam_prob <= thresholds["ambiguous_high"]:
        return True

    # 그 외 구간: 보수적으로 EXAONE 호출
    return True


# ==================== FastAPI 엔드포인트 ====================
class GateRequest(BaseModel):
    """게이트웨이 요청 모델."""

    email_text: str  # 이메일 텍스트 (제목, 본문 등)
    max_length: Optional[int] = 512  # 최대 토큰 길이
    request_id: Optional[str] = None  # 요청 ID (상태 관리용, 선택사항)


class GateResponse(BaseModel):
    """게이트웨이 응답 모델."""

    spam_prob: float  # 스팸 확률 (0.0 ~ 1.0)
    ham_prob: float  # 정상 확률 (0.0 ~ 1.0)
    label: str  # "spam" 또는 "ham"
    confidence: str  # "low", "medium", "high"
    latency_ms: float  # 처리 시간 (밀리초)
    should_call_exaone: bool  # EXAONE Reader 호출 필요 여부
    request_id: Optional[str] = None  # 요청 ID (있는 경우)


@router.post("/gate", response_model=GateResponse)
async def spam_gate(request: GateRequest):
    """KoELECTRA 게이트웨이 API 엔드포인트."""
    try:
        # 스팸 확률 계산
        gate_result = predict_spam_probability(
            request.email_text, request.max_length, request.request_id
        )

        # EXAONE 호출 여부 결정
        should_call = should_call_exaone(gate_result["spam_prob"])

        return GateResponse(
            spam_prob=gate_result["spam_prob"],
            ham_prob=gate_result["ham_prob"],
            label=gate_result["label"],
            confidence=gate_result["confidence"],
            latency_ms=gate_result["latency_ms"],
            should_call_exaone=should_call,
            request_id=gate_result.get("request_id"),
        )

    except RuntimeError as e:
        error_msg = str(e)
        print(f"[ERROR] 게이트웨이 모델 로드 실패: {error_msg}")
        raise HTTPException(
            status_code=503,
            detail=f"게이트웨이 모델 로드 실패: {error_msg[:300]}",
        )
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] 게이트웨이 처리 실패: {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"게이트웨이 처리 중 오류가 발생했습니다: {error_msg[:300]}",
        )


# ==================== 상태 관리 엔드포인트 ====================
@router.get("/gate/state/{request_id}")
async def get_gate_state(request_id: str):
    """저장된 게이트웨이 결과를 조회합니다.

    Args:
        request_id: 요청 ID

    Returns:
        게이트웨이 결과
    """
    if request_id not in _gate_results_cache:
        raise HTTPException(
            status_code=404,
            detail=f"요청 ID '{request_id}'에 대한 결과를 찾을 수 없습니다.",
        )

    return _gate_results_cache[request_id]


@router.delete("/gate/state/{request_id}")
async def delete_gate_state(request_id: str):
    """저장된 게이트웨이 결과를 삭제합니다.

    Args:
        request_id: 요청 ID

    Returns:
        삭제 성공 메시지
    """
    if request_id not in _gate_results_cache:
        raise HTTPException(
            status_code=404,
            detail=f"요청 ID '{request_id}'에 대한 결과를 찾을 수 없습니다.",
        )

    del _gate_results_cache[request_id]
    return {"message": f"요청 ID '{request_id}'의 결과가 삭제되었습니다."}


@router.get("/gate/state")
async def list_gate_states():
    """저장된 모든 게이트웨이 결과의 ID 목록을 반환합니다.

    Returns:
        요청 ID 목록
    """
    return {"request_ids": list(_gate_results_cache.keys())}


# ==================== EXAONE 툴 통합 엔드포인트 ====================
class SpamAnalyzeRequest(BaseModel):
    """스팸 분석 요청 모델."""

    email_text: str  # 이메일 텍스트


class SpamAnalyzeResponse(BaseModel):
    """스팸 분석 응답 모델."""

    gate_result: dict  # KoELECTRA 게이트웨이 결과
    exaone_result: Optional[str]  # EXAONE Reader 결과 (None일 수 있음)
    final_decision: str  # 최종 결정


@router.post("/spam-analyze", response_model=SpamAnalyzeResponse)
async def spam_analyze_with_tool(request: SpamAnalyzeRequest):
    """EXAONE 툴을 사용한 스팸 분석 API 엔드포인트.

    KoELECTRA 게이트웨이로 1차 필터링 후,
    애매한 경우 EXAONE 툴을 호출하여 정밀 검사를 수행합니다.
    """
    try:
        from app.service.verdict_agent import get_exaone_tool

        # 1단계: KoELECTRA 게이트웨이로 스팸 확률 계산
        gate_result = predict_spam_probability(request.email_text)
        spam_prob = gate_result["spam_prob"]

        print(f"[GATE] 스팸 확률: {spam_prob:.4f}, 레이블: {gate_result['label']}")

        # 2단계: EXAONE 호출 여부 결정
        should_call = should_call_exaone(spam_prob)

        exaone_result = None
        if should_call:
            # EXAONE 툴 호출
            print("[INFO] EXAONE 툴 호출 중...")
            exaone_tool = get_exaone_tool()
            exaone_result = exaone_tool.invoke(
                {
                    "email_text": request.email_text,
                    "spam_prob": spam_prob,
                    "label": gate_result["label"],
                    "confidence": gate_result["confidence"],
                }
            )
            print("[EXAONE] 정밀 검사 완료")

        # 3단계: 최종 결정 생성
        if exaone_result:
            final_decision = f"""KoELECTRA 게이트웨이 결과:
- 스팸 확률: {spam_prob:.4f}
- 레이블: {gate_result["label"]}
- 신뢰도: {gate_result["confidence"]}

EXAONE Reader 정밀 검사:
{exaone_result}

최종 판단: {"스팸으로 판단됩니다" if spam_prob > 0.5 else "정상 메일로 판단됩니다"}"""
        else:
            final_decision = f"""KoELECTRA 게이트웨이 결과:
- 스팸 확률: {spam_prob:.4f}
- 레이블: {gate_result["label"]}
- 신뢰도: {gate_result["confidence"]}

최종 판단: {"스팸으로 판단됩니다" if spam_prob > 0.5 else "정상 메일로 판단됩니다"} (EXAONE 호출 없음)"""

        return SpamAnalyzeResponse(
            gate_result=gate_result,
            exaone_result=exaone_result,
            final_decision=final_decision,
        )

    except RuntimeError as e:
        error_msg = str(e)
        print(f"[ERROR] 모델 로드 실패: {error_msg}")
        raise HTTPException(
            status_code=503,
            detail=f"모델 로드 실패: {error_msg[:300]}",
        )
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] 스팸 분석 실패: {error_msg}")
        import traceback

        print(f"[ERROR] 상세 오류:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"스팸 분석 중 오류가 발생했습니다: {error_msg[:300]}",
        )
