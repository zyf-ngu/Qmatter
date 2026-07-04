#!/usr/bin/env python3
"""
Agent工具调用实例 - 使用DeepSeek API
核心功能：展示工具创建、注入大模型、以及模型返回的tool_calls数据结构
"""

import json
import sys
from typing import List, Dict, Any
from openai import OpenAI
from config import api_key


# 从配置文件获取API配置
API_KEY = api_key
BASE_URL = "https://api.deepseek.com/v1"

if not API_KEY:
    print("错误: config.json中未提供api_key")
    sys.exit(1)

# 初始化OpenAI客户端（兼容DeepSeek API）
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ========== 1. 工具创建 ==========
print("=" * 60)
print("步骤1: 工具创建 - 定义工具函数和对应的JSON Schema")
print("=" * 60)

# ========== 2. 模拟工具的执行函数 ==========
def execute_weather(location: str, unit: str = "celsius") -> str:
    """模拟获取天气的函数"""
    # 模拟不同城市的天气数据
    weather_data = {
        "北京": "晴天，温度25°C，湿度45%",
        "上海": "多云，温度22°C，湿度70%",
        "深圳": "阵雨，温度28°C，湿度85%",
        "杭州": "阴天，温度20°C，湿度60%",
    }
    weather = weather_data.get(location, f"晴转多云，温度23°C")
    if unit == "fahrenheit":
        # 简单的摄氏度转华氏度
        return f"{location}天气: {weather} (华氏度)"
    return f"{location}天气: {weather}"


def execute_calculator(expression: str) -> str:
    """安全执行计算器"""
    # 注意：演示环境使用eval，生产环境建议使用更安全的方式（如ast.literal_eval或第三方库）
    # 这里仅为演示目的，限制了表达式长度和字符
    try:
        # 简单的安全检查：只允许数字、运算符、括号和空格
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "错误: 表达式包含非法字符"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def get_weather_tool() -> Dict[str, Any]:
    """创建获取天气的工具定义"""
    return {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、深圳",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认为摄氏度",
                    },
                },
                "required": ["location"],
            },
        },
    }


def calculator_tool() -> Dict[str, Any]:
    """创建计算器的工具定义"""
    return {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "进行数学计算，支持加减乘除和括号运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：'2+3*4', '(10-5)/2'",
                    },
                },
                "required": ["expression"],
            },
        },
    }


# 工具列表
tools = [get_weather_tool(), calculator_tool()]

# 展示创建的工具定义
print("创建的工具定义 (JSON Schema):")
for tool in tools:
    print(json.dumps(tool, indent=2, ensure_ascii=False))
    print("-" * 40)


def execute_tool_call(tool_call) -> str:
    """根据tool_call执行对应的工具函数"""
    func_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if func_name == "get_current_weather":
        location = arguments.get("location")
        unit = arguments.get("unit", "celsius")
        return execute_weather(location, unit)
    elif func_name == "calculator":
        expression = arguments.get("expression")
        return execute_calculator(expression)
    else:
        return f"未知工具: {func_name}"


# ========== 3. Agent主循环 ==========
def run_agent(user_query: str):
    """运行agent，展示工具注入和tool_calls结构"""
    print("\n" + "=" * 60)
    print(f"用户问题: {user_query}")
    print("=" * 60)

    # 初始化对话消息
    messages = [
        {
            "role": "system",
            "content": "你是一个有用的助手，可以使用提供的工具来回答用户问题。"
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    # ========== 4. 注入工具并调用大模型（第一次） ==========
    print("\n步骤2: 注入工具到大模型 - 第一次调用API，携带tools参数")
    print("-" * 60)
    print("请求参数中的tools (注入的工具定义):")
    print(json.dumps(tools, indent=2, ensure_ascii=False))
    print("-" * 60)
    print("正在调用DeepSeek模型...")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # 让模型自动决定是否调用工具
        temperature=0.7,
    )

    # 获取模型返回的消息
    assistant_message = response.choices[0].message

    # ========== 5. 展示模型返回的tool_calls数据结构 ==========
    print("\n步骤3: 大模型返回的tool_calls数据结构")
    print("-" * 60)

    if assistant_message.tool_calls:
        print("模型决定调用以下工具，tool_calls内容:")
        for i, tool_call in enumerate(assistant_message.tool_calls):
            print(f"\n工具调用 {i + 1}:")
            print(f"  - 工具ID: {tool_call.id}")
            print(f"  - 函数名: {tool_call.function.name}")
            print(f"  - 参数(JSON字符串): {tool_call.function.arguments}")
            print(f"  - 解析后的参数: {json.loads(tool_call.function.arguments)}")
        print("\n完整的tool_calls对象结构:")
        print(json.dumps(
            [{
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            } for tc in assistant_message.tool_calls],
            indent=2
        ))
    else:
        print("模型没有返回任何tool_calls，直接回答:")
        print(assistant_message.content)
        return

    # 将模型的回复添加到消息列表
    messages.append(assistant_message)

    # ========== 6. 执行工具调用 ==========
    print("\n步骤4: 执行工具调用，并将结果返回给模型")
    print("-" * 60)
    for tool_call in assistant_message.tool_calls:
        print(f"执行工具: {tool_call.function.name}")
        result = execute_tool_call(tool_call)
        print(f"执行结果: {result}")

        # 将工具执行结果作为tool消息添加到对话中
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # ========== 7. 再次调用模型，获取最终回答 ==========
    print("\n步骤5: 将工具执行结果反馈给模型，获取最终回答")
    print("-" * 60)
    second_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
    )
    final_answer = second_response.choices[0].message.content
    print("最终回答:")
    print(final_answer)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 示例1: 天气查询（会触发工具调用）
    print("\n【演示1：天气查询】")
    run_agent("请问上海今天的天气怎么样？")

    print("\n\n【演示2：数学计算】")
    run_agent("帮我计算一下 (15 + 7) * 3 的结果")

    # 可以取消下面注释来运行交互模式
    # while True:
    #     query = input("\n请输入你的问题 (输入exit退出): ")
    #     if query.lower() == 'exit':
    #         break
    #     run_agent(query)