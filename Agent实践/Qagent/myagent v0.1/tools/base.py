from typing import Any, Dict, List, Callable, Optional


class Tool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema for the function parameters
        self.func = func

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    async def execute(self, **kwargs) -> str:
        """调用工具函数，返回字符串结果（成功时为返回值，失败时为异常信息）"""
        try:
            result = await self.func(**kwargs)
            return str(result) if result is not None else "Success"
        except Exception as e:
            return f"Error executing tool: {e}"