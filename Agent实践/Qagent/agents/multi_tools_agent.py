# 1. 定义一个工具（计算器）
def calculator(expr: str) -> str:
    try:
        return str(eval(expr))
    except:
        return "计算错误"


# 2. Mock 的 LLM 决策函数（根据对话历史模拟思考）
def mock_llm(messages):
    # 检查最后几条消息，决定下一步动作
    history_str = str(messages)

    # 需要两次调用：先算 1+2，再算 3*3
    if "1+2" in history_str and "calculator" in history_str and "3*3" not in history_str:
        # 已经有 1+2 的结果，接下来调用计算 3*3
        return {"type": "tool_call", "name": "calculator", "args": {"expr": "3*3"}}
    elif "1+2" in history_str and "calculator" not in history_str:
        # 第一次调用：计算 1+2
        return {"type": "tool_call", "name": "calculator", "args": {"expr": "1+2"}}
    else:
        # 工具都调用完了，返回最终答案
        return {"type": "answer", "content": f"最终计算结果：{messages[-1]['content']}"}


# 3. Agent 主循环
def run_agent(user_query):
    messages = [{"role": "user", "content": user_query}]

    while True:
        decision = mock_llm(messages)

        if decision["type"] == "answer":
            print(f"最终回答: {decision['content']}")
            return decision["content"]

        # 执行工具调用
        tool_name = decision["name"]
        args = decision["args"]
        result = calculator(**args)
        print(f"[调用工具] {tool_name}({args}) -> {result}")

        # 将工具结果加入对话历史
        messages.append({"role": "ai", "content": decision})
        messages.append({"role": "tool", "content": result})


# 4. 运行
if __name__ == "__main__":
    run_agent("请计算 (1+2) 的结果，然后将结果乘以 3")