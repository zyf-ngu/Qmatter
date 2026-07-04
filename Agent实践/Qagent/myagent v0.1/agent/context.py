from typing import List, Dict, Any
from memory.manager import MemoryManager
from tools.registry import ToolRegistry

SYSTEM_PROMPT = """
你是一个个人 AI 助手，运行在用户的本地工作区。
你可以使用工具来读取文件或列出目录内容。
回答用户问题时请简洁准确，需要文件内容时主动使用工具。
"""


class ContextBuilder:
    def __init__(self, memory: MemoryManager, tool_registry: ToolRegistry):
        self.memory = memory
        self.tool_registry = tool_registry

    async def build(self) -> List[Dict[str, Any]]:
        # 系统提示
        system_content = SYSTEM_PROMPT
        long_term = await self.memory.get_long_term_context()
        if long_term:
            system_content += f"\n\n[Long-term Memory]\n{long_term}"

        messages = [{"role": "system", "content": system_content}]

        # 添加短期记忆中的历史对话
        history = await self.memory.get_context_messages()
        messages.extend(history)

        return messages