import os
from typing import List, Dict, Optional
from .pdf_parser import PDFParser
from .docx_parser import DocxParser
from .excel_parser import ExcelParser
from .txt_parser import TxtParser


class DocumentLoader:
    """文档加载器，统一解析各类文档"""
    
    def __init__(self):
        self.parsers = {
            '.pdf': PDFParser(),
            '.docx': DocxParser(),
            '.doc': DocxParser(),
            '.xlsx': ExcelParser(),
            '.xls': ExcelParser(),
            '.txt': TxtParser(),
            '.md': TxtParser()
        }
    
    def parse_file(self, file_path: str) -> Optional[Dict]:
        """解析单个文件，返回包含文件路径和文本内容的字典"""
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.parsers:
            print(f"不支持的文件格式: {ext}")
            return None
        
        try:
            texts = self.parsers[ext].parse(file_path)
            return {
                "file_path": file_path,
                "content": texts
            }
        except Exception as e:
            print(f"解析文件失败 {file_path}: {e}")
            return None
    
    def load_files(self, file_paths: List[str]) -> List[Dict]:
        """解析多个文件，返回包含文件路径和文本内容的列表"""
        results = []
        for path in file_paths:
            result = self.parse_file(path)
            if result:
                results.append(result)
        return results
    
    def load_directory(self, dir_path: str, recursive: bool = False) -> List[Dict]:
        """解析整个目录下的文档，返回包含文件路径和文本内容的列表"""
        if not os.path.isdir(dir_path):
            print(f"目录不存在: {dir_path}")
            return []
        
        results = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.parsers:
                    file_path = os.path.join(root, file)
                    result = self.parse_file(file_path)
                    if result:
                        results.append(result)
            if not recursive:
                break
        return results


__all__ = [
    "PDFParser",
    "DocxParser",
    "ExcelParser",
    "TxtParser",
    "DocumentLoader"
]
