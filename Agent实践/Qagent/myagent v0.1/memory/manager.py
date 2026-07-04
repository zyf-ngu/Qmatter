from typing import List, Dict, Any
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.base import BaseMemory


class MemoryManager(BaseMemory):
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    async def add_user_message(self, msg: str):
        self.short_term.add("user", msg)

    async def add_assistant_message(self, msg: str):
        self.short_term.add("assistant", msg)

    async def get_context_messages(self) -> List[Dict[str, Any]]:
        return self.short_term.get_messages()

    async def get_long_term_context(self) -> str:
        return self.long_term.read()