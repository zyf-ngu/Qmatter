class MockMemoryAgent:
    """
    模拟一个具有记忆和多轮状态的智能体。
    - 记忆：保存所有交互历史（用户输入和系统输出）
    - 状态：维护一个可自定义的状态字典，用于跨轮次跟踪信息
    """
    def __init__(self, initial_state=None):
        # 记忆列表：存储每一轮的 (用户输入, 系统输出)
        self.memory = []
        # 状态字典：用于存储多轮状态变量（如计数器、标志、用户偏好等）
        self.state = initial_state if initial_state is not None else {}
        # 可选：自定义响应生成函数，默认使用简单的规则
        self.response_generator = self._mock_llm_response

    def _mock_llm_response(self, user_input):
        """默认响应生成器：基于输入内容和当前状态返回模拟响应"""
        # 示例状态：记录用户问候次数
        if 'greeting_count' not in self.state:
            self.state['greeting_count'] = 0

        # 简单的模式匹配
        user_lower = user_input.lower()
        if '你好' in user_lower or 'hello' in user_lower:
            self.state['greeting_count'] += 1
            if self.state['greeting_count'] == 1:
                return "你好！很高兴见到你。"
            else:
                return f"我们又见面了！这是第 {self.state['greeting_count']} 次问候。"
        elif '天气' in user_lower:
            return "今天天气晴朗，适合出门。"
        elif '状态' in user_lower:
            # 展示当前状态（演示用）
            return f"当前状态: {self.state}"
        elif '重置' in user_lower:
            # 重置所有状态（记忆不清除，但状态重置）
            self.state = {}
            return "状态已重置。"
        elif '记忆' in user_lower:
            # 返回最近三条记忆
            recent = self.memory[-3:] if self.memory else []
            return f"最近的记忆: {recent}"
        elif '退出' in user_lower:
            return "再见！"
        else:
            return f"我不太理解 '{user_input}'，请再试一次。"

    def process_input(self, user_input):
        """处理用户输入：更新记忆，调用响应生成器，返回响应"""
        # 调用响应生成器获取回复
        response = self.response_generator(user_input)
        # 将本次交互存入记忆
        self.memory.append((user_input, response))
        return response


# 示例使用
if __name__ == "__main__":
    agent = MockMemoryAgent(initial_state={"step": 0})

    # 模拟多轮对话
    inputs = [
        "你好",
        "今天天气怎么样",
        "你好",
        "状态",
        "你好",
        "记忆",
        "重置",
        "状态"
    ]

    for user_input in inputs:
        response = agent.process_input(user_input)
        print(f"用户: {user_input}")
        print(f"Agent: {response}\n")