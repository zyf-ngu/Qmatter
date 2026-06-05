from typing import List
import os


class PDFParser:
    """PDF 文档解析器"""
    
    def __init__(self):
        self.parser_type = "pymupdf"  # 默认使用 PyMuPDF
    
    def parse(self, file_path: str) -> List[str]:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            texts = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    texts.append(text)
            doc.close()
            return texts
        except ImportError:
            try:
                import pdfplumber
                texts = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text and text.strip():
                            texts.append(text)
                return texts
            except ImportError:
                try:
                    import pypdf
                    texts = []
                    reader = pypdf.PdfReader(file_path)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text and text.strip():
                            texts.append(text)
                    return texts
                except Exception as e:
                    print(f"PDF 解析失败: {e}")
                    return []


if __name__ == "__main__":
    parser = PDFParser()
    # result = parser.parse("example.pdf")
    # print(result)
