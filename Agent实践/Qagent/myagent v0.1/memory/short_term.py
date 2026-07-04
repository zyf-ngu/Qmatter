from typing import List, Dict, Any, Optional


class ShortTermMemory:
    def __init__(self, max_messages: int = 20):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages

    def add(self, role: str, content: Optional[str], **extra):
        msg = {"role": role, "content": content}
        msg.update(extra)
        self.messages.append(msg)
        # 保持最多 max_messages 条
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> List[Dict[str, Any]]:
        return self.messages