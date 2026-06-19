from typing import List


class TxtParser:
    """纯文本文件解析器"""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def parse(self, file_path: str) -> List[str]:
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                text = f.read()
            return [text] if text.strip() else []
        except Exception as e:
            print(f"TXT 解析失败: {e}")
            return []


if __name__ == "__main__":
    parser = TxtParser()
    # result = parser.parse("example.txt")
    # print(result)
