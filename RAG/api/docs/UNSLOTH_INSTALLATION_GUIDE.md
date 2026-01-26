# Unsloth 설치 가이드

## 설치 실패 원인

현재 네트워크 연결 문제로 자동 설치가 실패했습니다.

---

## 수동 설치 방법

### 방법 1: 기본 설치 (권장)

```bash
pip install unsloth
```

### 방법 2: GitHub에서 직접 설치

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 방법 3: 최신 버전 설치

```bash
pip install --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

## 설치 후 확인

```bash
python -c "import unsloth; print(unsloth.__version__)"
```

---

## 호환성 테스트 실행

설치 완료 후:

```bash
cd api
python scripts/test_unsloth_compatibility.py
```

---

## 네트워크 문제 해결

### 프록시 설정 (필요 시)

```bash
pip install --proxy http://proxy.example.com:8080 unsloth
```

### 오프라인 설치 (대안)

1. 다른 환경에서 Unsloth 다운로드
2. wheel 파일로 설치

---

## 참고

- Unsloth는 GitHub에서 직접 설치해야 할 수 있음
- 네트워크 연결이 안정적인 환경에서 설치 권장
- 설치 후 호환성 테스트 필수
