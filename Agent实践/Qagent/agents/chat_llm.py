import time


class MockLLM:
    """模拟 LLM 的回复，仅用于演示"""
    def invoke(self, prompt: str) -> str:
        print(f"[LLM] 收到提示: {prompt[:50]}...")
        time.sleep(0.1) # 模拟网络延迟
        # 极简规则回复
        if "天气" in prompt:
            return "当前天气晴朗，气温 25℃。"
        elif "你好" in prompt:
            return "你好！有什么可以帮助你的吗？"
        else:
            return "我不太理解你的问题，请换一种方式描述。"


def basic_chat():
    llm = MockLLM()
    print("=== 基础对话模型 (无状态) ===")
    while True:
        user_input = input("用户: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = llm.invoke(user_input)
        print(f"AI: {response}")


if __name__ == "__main__":
    basic_chat()