# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 루트 디렉토리의 requirements.txt 복사 (build context가 루트이므로)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 복사 (build context가 루트이므로)
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]