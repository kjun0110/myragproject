"""
PDF 추출 전략 열거형.

KoELECTRA 분류 모델의 출력 레이블(0~6)과 전략 구현체를 매핑합니다.
전략 패턴의 Context가 이 enum으로 선택된 전략을 결정합니다.
"""

from enum import IntEnum


class PDFStrategyLabel(IntEnum):
    """
    KoELECTRA 레이블 → 전략 매핑 (pdf_strategy.md 기준).

    | Label | 전략 | 주요 선택 기준 |
    | 0 | PyMuPDF | 텍스트 위주, 빠른 처리, 레이아웃 단순 |
    | 1 | pdfplumber | 표 포함, 선 기반 데이터 추출 |
    | 2 | pdfminer.six | 정교한 위치/폰트 분석 |
    | 3 | pypdf | 단순 텍스트 추출 또는 병합/분할 연계 |
    | 4 | LlamaParse | 복잡한 계층, Markdown, RAG 최적화 |
    | 5 | AWS Textract | 스캔 이미지, 영수증/청구서, 고정밀 OCR |
    | 6 | Google Document AI | 필기체, 다국어, 구글 클라우드 연동 |
    """

    PY_MU_PDF = 0
    PDF_PLUMBER_PDF = 1
    PDF_MINER_SIX = 2
    PY_PDF = 3
    LLAMA_PARSE = 4
    AWS_TEXTRACT = 5
    GOOGLE_DOCUMENT = 6

    @property
    def module_name(self) -> str:
        """동적 로드용 서브모듈 이름 (strategy_imples/pdf/ 하위)."""
        return _MODULE_MAP[self.value]

    @property
    def class_name(self) -> str:
        """해당 모듈 내 전략 클래스 이름."""
        return _CLASS_MAP[self.value]

    @classmethod
    def from_koelectra_label(cls, label: int) -> "PDFStrategyLabel":
        """KoELECTRA 모델 출력(0~6)을 전략 enum으로 변환."""
        try:
            return cls(label)
        except ValueError:
            return cls.PY_MU_PDF  # 기본값: 빠른 텍스트 추출


_MODULE_MAP: dict[int, str] = {
    0: "py_mu_pdf",
    1: "pdf_plumber_pdf",
    2: "pdf_miner_six",
    3: "py_pdf",
    4: "llama_parse",
    5: "aws_textract",
    6: "google_document",
}

_CLASS_MAP: dict[int, str] = {
    0: "PyMuPDFExtractionStrategy",
    1: "PdfPlumberExtractionStrategy",
    2: "PdfMinerSixExtractionStrategy",
    3: "PyPDFExtractionStrategy",
    4: "LlamaParseExtractionStrategy",
    5: "AWSTextractExtractionStrategy",
    6: "GoogleDocumentExtractionStrategy",
}


def get_strategy_for_label(label: int):
    """
    KoELECTRA 라벨(0~6)에 해당하는 PDF 추출 전략 인스턴스를 동적 로드하여 반환.

    LangGraph Router Node에서 KoELECTRA 출력 → 전략 실행 시 사용.

    Returns:
        PDFExtractionStrategy 구현체 인스턴스
    """
    import importlib

    strategy_label = PDFStrategyLabel.from_koelectra_label(label)
    module = importlib.import_module(
        f"app.domains.shared.strategy_imples.pdf.{strategy_label.module_name}"
    )
    strategy_class = getattr(module, strategy_label.class_name)
    return strategy_class()
