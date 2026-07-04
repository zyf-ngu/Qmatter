from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        返回格式：
        {
            "content": str | None,
            "tool_calls": [ {"name": "...", "arguments": {...}} ] | None
        }
        """
        pass