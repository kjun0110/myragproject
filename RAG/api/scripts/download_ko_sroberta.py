"""
jhgan/ko-sroberta-multitask 모델을 Hugging Face 캐시에 다운로드합니다.

실행 (프로젝트 루트 RAG/ 또는 api/ 에서):
  python api/scripts/download_ko_sroberta.py

캐시 위치 (기본):
  Windows: C:\\Users\\<USER>\\.cache\\huggingface\\hub\\
  또는 환경변수 HF_HOME, HUGGINGFACE_HUB_CACHE 로 지정 가능
"""
import os
import sys
from pathlib import Path

# api 경로 추가 (프로젝트 루트에서 실행 시)
_api_dir = Path(__file__).resolve().parent.parent
if str(_api_dir) not in sys.path:
    sys.path.insert(0, str(_api_dir))


def main():
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        print(f"캐시 디렉터리(env): {cache_dir}")
    else:
        default = Path.home() / ".cache" / "huggingface" / "hub"
        print(f"캐시 디렉터리(기본): {default}")

    repo_id = "jhgan/ko-sroberta-multitask"
    print(f"다운로드 중: {repo_id} ...")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=repo_id)
        print(f"완료. 저장 위치: {path}")
        return 0
    except ImportError:
        pass

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(repo_id)
        print("완료. (SentenceTransformer 로 캐시에 저장됨)")
        return 0
    except ImportError:
        pass

    try:
        from transformers import AutoModel, AutoTokenizer
        AutoTokenizer.from_pretrained(repo_id)
        AutoModel.from_pretrained(repo_id)
        print("완료. (transformers 로 캐시에 저장됨)")
        return 0
    except ImportError:
        pass

    print("실패: huggingface_hub, sentence_transformers, transformers 중 하나가 필요합니다.")
    print("  pip install huggingface_hub")
    return 1


if __name__ == "__main__":
    sys.exit(main())
