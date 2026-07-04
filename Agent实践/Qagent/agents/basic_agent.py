import json

from typing import Dict, Any, List


class ToolExecutor:
    """工具执行器（Mock）"""

    @staticmethod
    def get_weather(city: str) -> str:
        return f"{city}当前晴朗，22°C，湿度40%"

    @staticmethod
    def calculator(expression: str) -> str:
        # 仅用于演示，实际应使用安全的 eval 或自定义解析
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except:
            return "计算表达式无效"

    def execute(self, tool_name: str, arguments: Dict) -> str:
        if tool_name == "get_weather":
            return self.get_weather(**arguments)
        elif tool_name == "calculator":
            return self.calculator(**arguments)
        else:
            return f"未知工具: {tool_name}"


class MockToolLLM:
    """模拟支持工具调用的 LLM"""

    def invoke_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        user_msg = messages[-1]["content"]
        print(f"[LLM] 分析用户输入: {user_msg}")
        # 模拟决策：检测关键词决定是否调用工具
        if "天气" in user_msg:
            # 所谓支持工具调用，就是大模型返回调用函数的参数等信息
            return {"tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]}
        elif "计算" in user_msg:  # 简单提取数字，此处仅演示
            return {"tool_calls": [{"name": "calculator", "arguments": {"expression": "3+5"}}]}
        else:
            return {"content": "我是一个助手，可以帮你查询天气或计算。"}


def tool_agent_demo():
    llm = MockToolLLM()
    executor = ToolExecutor()
    tools = [{"name": "get_weather", "description": "获取指定城市的天气", "parameters": {"city": "string"}},
             {"name": "calculator", "description": "执行数学计算", "parameters": {"expression": "string"}}]
    print("=== Agent + 工具调用 ===")
    user_input = input("用户: ")
    messages = [{"role": "user", "content": user_input}]
    # 第一次 LLM 调用
    response = llm.invoke_with_tools(messages, tools)
    if "tool_calls" in response:
        tool_call = response["tool_calls"][0]
        tool_name = tool_call["name"]
        args = tool_call["arguments"]
        print(f"[Agent] 决定调用工具: {tool_name}({args})")
        # 执行工具
        tool_result = executor.execute(tool_name, args)
        print(f"[工具返回]: {tool_result}")
        # 将工具结果作为新消息再次给 LLM
        messages.append({"role": "assistant", "tool_calls": [tool_call]})
        messages.append({"role": "tool", "content": tool_result})
        # 第二次 LLM 调用（生成最终回答）
        final_response = llm.invoke_with_tools(messages, tools)
        print(f"AI: {final_response.get('content', '操作完成')}")
    else:
        print(f"AI: {response.get('content')}")


if __name__ == "__main__":
    tool_agent_demo()
