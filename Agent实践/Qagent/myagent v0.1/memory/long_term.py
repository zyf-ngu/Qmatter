from pathlib import Path
from config import settings


class LongTermMemory:
    def __init__(self):
        self.memory_path = Path(settings.workspace_dir) / settings.memory_file

    def read(self) -> str:
        """读取 MEMORY.md，如果不存在则创建并返回提示"""
        if self.memory_path.is_file():
            try:
                return self.memory_path.read_text(encoding='utf-8')
            except Exception:
                return ""
        else:
            # 确保目录存在并创建默认文件
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            default_content = "# Long-term Memory\n\nWrite important facts, preferences here.\n"
            self.memory_path.write_text(default_content, encoding='utf-8')
            return default_content

    def write(self, content: str):
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(content, encoding='utf-8')