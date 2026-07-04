import json
from typing import Optional
from llm.base import BaseLLM
from memory.manager import MemoryManager
from tools.registry import ToolRegistry
from bus.message_bus import Message, MessageBus
from agent.context import ContextBuilder

MAX_ITERATIONS = 10  # 防止无限工具调用循环


class AgentLoop:
    def __init__(self, llm: BaseLLM, tool_registry: ToolRegistry, memory: MemoryManager, bus: MessageBus):
        self.llm = llm
        self.tool_registry = tool_registry
        self.memory = memory
        self.bus = bus
        self.context_builder = ContextBuilder(memory, tool_registry)

    async def handle_message(self, msg: Message) -> Optional[str]:
        # 将用户消息存储到记忆
        await self.memory.add_user_message(msg.text)

        # 工具调用循环
        iterations = 0
        while iterations < MAX_ITERATIONS:
            iterations += 1
            # 构建完整上下文
            messages = await self.context_builder.build()
            tools = self.tool_registry.get_tool_schemas()

            # 调用 LLM
            response = await self.llm.generate(messages, tools)

            if response.get("tool_calls"):
                # 构建带 tool_calls 的 assistant 消息（API 要求的格式）
                assistant_tool_calls = []
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    arguments_str = tc["arguments"] if isinstance(tc["arguments"], str) else json.dumps(tc["arguments"])
                    assistant_tool_calls.append({
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments_str,
                        }
                    })
                # 存储一条带 tool_calls 的 assistant 消息
                self.memory.short_term.add("assistant", None, tool_calls=assistant_tool_calls)

                # 执行每个工具并存储结果（tool 消息必须带 tool_call_id）
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    try:
                        arguments = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                    except json.JSONDecodeError:
                        arguments = {}
                    print(f"🔧 执行工具: {tool_name} 参数: {arguments}")
                    result = await self.tool_registry.execute(tool_name, arguments)
                    print(f"📄 工具结果: {result[:200]}...")
                    self.memory.short_term.add("tool", result, tool_call_id=tc.get("id"))
                continue  # 重新让 LLM 处理结果

            if response.get("content"):
                # 得到文本回复
                reply_text = response["content"]
                await self.memory.add_assistant_message(reply_text)
                return reply_text

        return "抱歉，处理你的请求时遇到了问题（达到最大循环次数）。"