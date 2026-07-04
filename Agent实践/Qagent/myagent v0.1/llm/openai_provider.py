from typing import Any, Dict, List
from openai import AsyncOpenAI
from llm.base import BaseLLM
from config import settings


class OpenAIProvider(BaseLLM):
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model

    async def generate(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
        )
        msg = response.choices[0].message
        result = {"content": msg.content, "tool_calls": None}

        if msg.tool_calls:
            result["tool_calls"] = []
            for tc in msg.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,  # 保留为字符串，在循环中解析
                })
        return result