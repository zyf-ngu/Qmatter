import os
import re
import json
from openai import OpenAI
from config import api_key

# ========== 1. DeepSeek API 调用封装 ==========
# 请设置环境变量 DEEPSEEK_API_KEY 或直接填入你的密钥
DEEPSEEK_API_KEY = api_key
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)


def call_deepseek(messages, temperature=0.0):
    """调用DeepSeek Chat模型"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ========== 2. ReAct 范式演示 ==========
# 问题：需要多次使用计算器工具的数学问题
# 工具定义：简单计算器
def calculator(expression: str) -> str:
    """安全地计算数学表达式（演示用）"""
    try:
        # 限制允许的字符，避免恶意代码
        if not re.match(r'^[\d\+\-\*\/\(\)\.\s]+$', expression):
            return "错误：表达式包含非法字符"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


def demo_react():
    print("=" * 60)
    print("ReAct 范式演示：思考 -> 行动 -> 观察 循环")
    print("=" * 60)

    # 需要多步计算的复杂问题
    question = "请计算 (15 + 3) * 2 / 6 的结果，然后再加 10，最后乘以 3。一步步来。"
    print(f"用户问题: {question}\n")

    # ReAct 提示模板，强制模型输出结构化步骤
    system_prompt = """
你是一个使用ReAct范式的智能体。解决以下任务时，请严格按照格式输出：

Thought: 你当前对问题的思考。
Action: 要执行的动作，可以是 Calculator 或 Final Answer。
Action Input: 动作的输入参数。

- 如果使用 Calculator，Action Input 应为数学表达式，例如 "3*4+2"。
- 如果得到最终答案，Action 应为 Final Answer，Action Input 为最终答案（数字）。

注意：每次只输出一个 Thought/Action/Action Input 组合。
"""

    # 对话历史
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    max_steps = 5
    step = 0

    while step < max_steps:
        step += 1
        print(f"--- 步骤 {step} ---")
        response = call_deepseek(messages, temperature=0.0)
        print(f"模型输出:\n{response}\n")

        # 解析 Action 和 Action Input
        action_match = re.search(r"Action:\s*(\w+)\s*\nAction Input:\s*(.+)", response, re.IGNORECASE)
        if not action_match:
            # 如果没有找到标准格式，尝试直接提取 Final Answer
            if "final answer" in response.lower():
                final_match = re.search(r"final answer.*?(\d+(?:\.\d+)?)", response, re.IGNORECASE)
                if final_match:
                    print(f"最终答案: {final_match.group(1)}")
                    break
            print("无法解析动作，终止循环。")
            break

        action = action_match.group(1).strip()
        action_input = action_match.group(2).strip()

        if action.lower() == "calculator":
            print(f"🔧 执行计算: {action_input}")
            observation = calculator(action_input)
            print(f"📊 观察结果: {observation}\n")
            # 将模型的思考过程和观察结果加入对话历史
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        elif action.lower() == "final answer":
            print(f"✅ 最终答案: {action_input}")
            break
        else:
            print(f"未知动作: {action}")
            break
    else:
        print("达到最大步数，未得到最终答案。")

    print("\n" + "=" * 60 + "\n")


# ========== 3. Plan-and-Solve 范式演示 ==========
def demo_plan_and_solve():
    print("=" * 60)
    print("Plan-and-Solve 范式演示：先生成计划，再逐步求解")
    print("=" * 60)

    question = "一个长方形的长是宽的2倍，周长是30厘米。请问这个长方形的面积是多少平方厘米？"
    print(f"用户问题: {question}\n")

    # 第一阶段：生成计划
    plan_prompt = f"""
请为以下问题制定一个详细的求解计划，要求分步骤列出（使用数字列表）。
问题：{question}

输出格式：
Plan:
1. 第一步...
2. 第二步...
3. ...
"""
    plan_messages = [{"role": "user", "content": plan_prompt}]
    plan_response = call_deepseek(plan_messages, temperature=0.0)
    print("【生成的计划】")
    print(plan_response)
    print("\n")

    # 提取计划步骤（简单正则匹配）
    steps = re.findall(r"\d+\.\s*(.+)", plan_response)
    if not steps:
        print("无法解析计划，使用默认步骤。")
        steps = ["设宽为x，则长为2x", "根据周长公式列出方程", "解出x", "计算面积"]

    # 第二阶段：按计划逐步求解
    context = f"问题：{question}\n计划：{plan_response}\n\n现在开始逐步执行计划：\n"
    results = []

    for i, step_desc in enumerate(steps, 1):
        print(f"--- 执行步骤 {i}: {step_desc} ---")
        step_prompt = f"{context}\n当前步骤：{step_desc}\n请根据已有信息完成这一步，输出计算结果或推导过程。如果这一步是计算，请给出数值结果。"
        step_messages = [{"role": "user", "content": step_prompt}]
        step_result = call_deepseek(step_messages, temperature=0.0)
        print(f"步骤输出:\n{step_result}\n")
        results.append(step_result)
        # 更新上下文，为下一步提供信息
        context += f"步骤{i}结果：{step_result}\n"

    # 最终总结答案
    final_prompt = f"{context}\n请根据以上所有步骤的结果，给出问题的最终答案（仅数字+单位）。"
    final_answer = call_deepseek([{"role": "user", "content": final_prompt}], temperature=0.0)
    print("🎯 最终答案:", final_answer)
    print("\n" + "=" * 60 + "\n")


# ========== 4. Reflection 范式演示 ==========
def demo_reflection():
    print("=" * 60)
    print("Reflection 范式演示：生成答案 -> 自我反思 -> 修正答案")
    print("=" * 60)

    # 经典易错问题（很多人会直觉回答 0.1 美元）
    question = "一个球拍和一个球总共1.10美元。球拍比球贵1.00美元。请问球多少钱？"
    print(f"用户问题: {question}\n")

    # 第一阶段：直接回答（不反思）
    direct_prompt = f"请直接回答以下问题，不需要解释过程：{question}"
    direct_answer = call_deepseek([{"role": "user", "content": direct_prompt}], temperature=0.0)
    print(f"【直接回答】\n{direct_answer}\n")

    # 第二阶段：反思
    reflection_prompt = f"""
你之前回答的问题是：{question}
你的回答是：{direct_answer}

请仔细反思你的答案是否正确。如果正确，请说明理由；如果错误，请指出错误原因并给出正确的答案和解释。要求以以下格式输出：

反思内容：(你的思考过程)
修正后的答案：(最终正确答案，数字)
"""
    reflection_response = call_deepseek([{"role": "user", "content": reflection_prompt}], temperature=0.0)
    print("【自我反思与修正】")
    print(reflection_response)

    # 提取最终修正的答案
    corrected_match = re.search(r"修正后的答案.*?(\d+\.?\d*)", reflection_response, re.IGNORECASE)
    if corrected_match:
        print(f"\n✅ 经过反思修正后的最终答案: {corrected_match.group(1)} 美元")
    else:
        print("\n⚠️ 未能从反思中提取到明确的修正答案。")

    print("\n" + "=" * 60 + "\n")


# ========== 5. 主程序 ==========
if __name__ == "__main__":
    # 检查API Key
    if DEEPSEEK_API_KEY == "your-api-key-here":
        print("错误：请先设置 DEEPSEEK_API_KEY 环境变量或直接填入有效密钥。")
        exit(1)

    # 运行三种范式演示
    demo_react()
    demo_plan_and_solve()
    demo_reflection()