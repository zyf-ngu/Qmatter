from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseMemory(ABC):
    @abstractmethod
    async def add_user_message(self, msg: str):
        pass

    @abstractmethod
    async def add_assistant_message(self, msg: str):
        pass

    @abstractmethod
    async def get_context_messages(self) -> List[Dict[str, Any]]:
        """返回用于 LLM 的上下文消息列表（不含系统提示）"""
        pass

    @abstractmethod
    async def get_long_term_context(self) -> str:
        """返回要注入系统提示的长期记忆文本"""
        pass