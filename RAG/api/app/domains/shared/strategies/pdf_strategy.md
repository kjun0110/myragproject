# PDF 추출 전략 분류 모델 (KoELECTRA) 설계 가이드

이 문서는 7개의 PDF 라이브러리 중 최적의 전략을 선택하기 위한 KoELECTRA 모델의 훈련 프롬프트 및 분류 기준을 정의합니다.

## 1. 전략별 분류 레이블 (Labels)
모델은 입력된 텍스트/특성을 분석하여 다음 7개 레이블 중 하나로 분류합니다.

| Label | 전략 클래스 (Implementation) | 주요 선택 기준 (Trigger) |
| :--- | :--- | :--- |
| **0** | `py_mu_pdf.py` | 텍스트 위주, 빠른 처리 속도 필요, 레이아웃 단순함 |
| **1** | `pdf_plumber.py` | 표(Table)가 포함됨, 선 기반의 데이터 추출 필요 |
| **2** | `pdf_miner_six.py` | 텍스트의 정교한 위치 정보(좌표) 및 폰트 분석 필요 |
| **3** | `py_pdf.py` | 매우 단순한 텍스트 추출 또는 단순 병합/분할 연계 시 |
| **4** | `llama_parse.py` | 복잡한 계층 구조, Markdown 변환 필요, RAG 최적화 데이터 |
| **5** | `aws_textract.py` | 스캔된 이미지, 정형 서식(영수증/청구서), 고정밀 OCR 필요 |
| **6** | `google_document.py` | 필기체 포함, 다국어 혼합, 구글 클라우드 에코시스템 연동 |

---

## 2. 훈련 데이터 생성을 위한 페르소나 (Prompting for Data Gen)
KoELECTRA 훈련을 위해 엑사원(EXAONE) 등에 아래 프롬프트를 입력하여 학습용 데이터셋(JSONL)을 생성하십시오.

> **Prompt:**
> "너는 PDF 데이터 엔지니어다. 다음 7가지 라이브러리(`PyMuPDF`, `pdfplumber`, `pdfminer.six`, `pypdf`, `LlamaParse`, `AWS Textract`, `Google Document AI`)가 필요한 각각의 상황을 한국어 문장으로 100개씩 생성해줘.
> 사용자의 요구사항 예시:
> - '표가 너무 많아서 데이터가 다 깨져요.' -> `pdfplumber`
> - '영수증 사진인데 글자를 읽어야 해요.' -> `aws_textract`
> - '속도가 제일 빨랐으면 좋겠어요.' -> `py_mu_pdf`
> 결과는 [문장, 레이블 번호] 형식의 JSONL로 출력해줘."

---

## 3. KoELECTRA 입력 데이터 구조 (Input Features)
모델의 정확도를 높이기 위해 텍스트 외에 다음 메타데이터를 토큰화하여 입력합니다.
- **Input Text:** "[File_Size: Low] [Page_Count: 10] 문서에 표가 많고 선이 복잡하게 얽혀 있어."
- **Expected Output:** `1` (pdf_plumber)

---

## 4. 모델 활용 전략 (In LangGraph)
1. **Router Node:** KoELECTRA가 PDF의 첫 500자나 메타데이터를 입력받아 Label을 출력합니다.
2. **Strategy Factory:** 출력된 Label에 매칭되는 `strategy_imples/pdf/` 내의 `.py` 파일을 동적으로 로드하여 실행합니다.
