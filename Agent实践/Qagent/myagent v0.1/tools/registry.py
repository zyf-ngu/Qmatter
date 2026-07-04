from typing import Dict, List, Any
from tools.base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Tool '{name}' not found."
        return await tool.execute(**arguments)


# 全局工具注册表
tool_registry = ToolRegistry()