#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""모델 파일 형식 확인 스크립트.

HuggingFace 캐시에서 실제로 사용되는 모델 파일 형식을 확인합니다.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_url
from huggingface_hub.utils import HfHubHTTPError

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
scripts_dir = current_file.parent  # api/scripts/
api_dir = scripts_dir.parent  # api/

sys.path.insert(0, str(api_dir))

from app.core.loaders import ModelLoader


def check_model_format(model_id: str):
    """모델 파일 형식을 확인합니다.
    
    Args:
        model_id: HuggingFace 모델 ID
    """
    print("=" * 60)
    print(f"모델 형식 확인: {model_id}")
    print("=" * 60)
    
    # HuggingFace 캐시 경로 확인
    cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
    if cache_dir:
        print(f"\n[INFO] HuggingFace 캐시 경로: {cache_dir}")
    else:
        default_cache = Path.home() / ".cache" / "huggingface"
        print(f"\n[INFO] HuggingFace 캐시 경로 (기본값): {default_cache}")
        cache_dir = str(default_cache)
    
    # 모델 캐시 경로
    model_cache_dir = Path(cache_dir) / "hub" / model_id.replace("/", "--")
    
    print(f"\n[INFO] 모델 캐시 디렉토리: {model_cache_dir}")
    
    # 캐시 디렉토리 확인
    if not model_cache_dir.exists():
        print(f"\n[WARNING] 모델 캐시 디렉토리가 없습니다.")
        print(f"[INFO] 모델을 다운로드하면 자동으로 생성됩니다.")
        return
    
    # 모델 파일 확인
    print(f"\n[INFO] 모델 파일 검색 중...")
    
    safetensors_files = list(model_cache_dir.rglob("*.safetensors"))
    bin_files = list(model_cache_dir.rglob("*.bin"))
    pt_files = list(model_cache_dir.rglob("*.pt"))
    pth_files = list(model_cache_dir.rglob("*.pth"))
    
    print(f"\n[결과] 파일 형식:")
    print(f"  - .safetensors 파일: {len(safetensors_files)}개")
    print(f"  - .bin 파일: {len(bin_files)}개")
    print(f"  - .pt 파일: {len(pt_files)}개")
    print(f"  - .pth 파일: {len(pth_files)}개")
    
    if safetensors_files:
        print(f"\n[✅] Safetensors 형식 발견!")
        print(f"  파일 목록:")
        for f in safetensors_files[:5]:  # 최대 5개만 표시
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.name} ({size_mb:.2f} MB)")
        if len(safetensors_files) > 5:
            print(f"    ... 외 {len(safetensors_files) - 5}개 파일")
    else:
        print(f"\n[⚠️] Safetensors 형식 없음!")
        print(f"[INFO] PyTorch pickle 형식(.bin)을 사용 중입니다.")
        print(f"[INFO] 이는 safetensors보다 느릴 수 있습니다.")
    
    if bin_files:
        print(f"\n[⚠️] PyTorch pickle 형식(.bin) 발견:")
        for f in bin_files[:5]:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    - {f.name} ({size_mb:.2f} MB)")
    
    # 모델 인덱스 파일 확인
    index_files = list(model_cache_dir.rglob("*index.json"))
    if index_files:
        print(f"\n[INFO] 모델 인덱스 파일:")
        for f in index_files:
            print(f"    - {f.name}")
            # 인덱스 파일 내용 확인
            try:
                import json
                with open(f, 'r', encoding='utf-8') as file:
                    index_data = json.load(file)
                    if 'weight_map' in index_data:
                        weight_map = index_data['weight_map']
                        safetensors_count = sum(1 for v in weight_map.values() if '.safetensors' in v)
                        bin_count = sum(1 for v in weight_map.values() if '.bin' in v)
                        print(f"      - Safetensors 가중치: {safetensors_count}개")
                        print(f"      - Bin 가중치: {bin_count}개")
            except Exception as e:
                print(f"      - 인덱스 파일 읽기 실패: {e}")
    
    # HuggingFace Hub에서 직접 확인
    print(f"\n[INFO] HuggingFace Hub에서 모델 정보 확인 중...")
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        
        # 모델 파일 목록
        files = [f.rfilename for f in info.siblings]
        safetensors_in_hub = [f for f in files if f.endswith('.safetensors')]
        bin_in_hub = [f for f in files if f.endswith('.bin')]
        
        print(f"  - Hub에 .safetensors 파일: {len(safetensors_in_hub)}개")
        print(f"  - Hub에 .bin 파일: {len(bin_in_hub)}개")
        
        if safetensors_in_hub:
            print(f"  ✅ Hub에 safetensors 형식 존재!")
            if not safetensors_files:
                print(f"  ⚠️ 하지만 로컬 캐시에는 없습니다.")
                print(f"  [INFO] 모델을 다시 다운로드하면 safetensors 형식으로 저장됩니다.")
        elif bin_in_hub:
            print(f"  ⚠️ Hub에 safetensors 형식이 없습니다.")
            print(f"  [INFO] 모델이 safetensors를 지원하지 않을 수 있습니다.")
    except Exception as e:
        print(f"  [WARNING] Hub 정보 확인 실패: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # EXAONE 모델 확인
    print("\n[1] EXAONE 모델 형식 확인")
    check_model_format(ModelLoader.EXAONE_MODEL_ID)
    
    # KoELECTRA 모델 확인
    print("\n[2] KoELECTRA 모델 형식 확인")
    check_model_format(ModelLoader.KOELECTRA_MODEL_ID)
