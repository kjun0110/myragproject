"""
[Label 5] AWS Textract 전략.

스캔된 이미지, 정형 서식(영수증/청구서), 고정밀 OCR이 필요할 때 선택.
"""

from pathlib import Path

from app.domains.shared.strategies.pdf_strategy import PDFExtractionStrategy


class AWSTextractExtractionStrategy(PDFExtractionStrategy):
    """AWS Textract 기반 PDF/이미지 OCR 추출."""

    def extract(self, file_path: str) -> str:
        try:
            import boto3
        except ImportError:
            raise RuntimeError("AWS Textract 전략 사용을 위해 pip install boto3 필요")

        path = Path(file_path)
        if not path.exists():
            return ""

        with open(path, "rb") as f:
            blob = f.read()

        client = boto3.client("textract")
        if file_path.lower().endswith(".pdf"):
            response = client.analyze_document(
                Document={"Bytes": blob},
                FeatureTypes=["TABLES", "FORMS"],
            )
        else:
            response = client.detect_document_text(Document={"Bytes": blob})

        blocks = response.get("Blocks", [])
        lines = []
        for block in blocks:
            if block.get("BlockType") == "LINE":
                text = block.get("Text", "")
                if text:
                    lines.append(text)
        return "\n".join(lines)
