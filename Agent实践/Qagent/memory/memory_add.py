import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional


class MemoryManager:
    """
    记忆管理器，演示：
    1. 原始存储 (ConversationBufferMemory 风格)
    2. 摘要压缩存储 (ConversationSummaryMemory 风格)
    3. 结构化提取存储 (ChatGPT 记忆风格)
    4. 触发写入机制
    """

    def __init__(self):
        # 原始存储：全量对话列表
        self.raw_messages: List[str] = []
        # 摘要存储：压缩后的对话摘要
        self.summary: str = ""
        # 结构化存储：用户画像 KV
        self.profile: Dict[str, Any] = {}
        # 写入策略配置
        self.auto_summary_rounds = 5   # 每 5 轮对话自动生成摘要
        self.pending_changes = False   # 标记是否有待同步的结构化更新

    # ------------------- 原始存储 -------------------
    def add_message(self, role: str, content: str):
        """直接追加原始对话，类似 ConversationBufferMemory"""
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] {role}: {content}"
        self.raw_messages.append(msg)
        print(f"[原始存储] 已记录: {msg}")

    # ------------------- 摘要压缩存储 -------------------
    def generate_summary(self, force: bool = False):
        """
        用 LLM 将全量对话压缩为摘要。
        此处用规则模拟，实际应调用 LLM 生成真实摘要。
        """
        if not self.raw_messages:
            self.summary = "暂无对话记录。"
            return

        # 模拟摘要生成：取最后一条用户消息 + 简单概括
        last_user_msg = next((m for m in reversed(self.raw_messages) if "user:" in m), "")
        if "requests" in last_user_msg.lower() or "爬虫" in last_user_msg.lower():
            self.summary = "用户正在进行 Python 爬虫相关讨论，涉及 requests 库使用。"
        elif "姓名" in last_user_msg or "我叫" in last_user_msg:
            self.summary = "用户在介绍个人基本信息。"
        else:
            self.summary = f"已进行 {len(self.raw_messages)} 轮对话，主题待抽取。"

        print(f"[摘要存储] 摘要已更新: {self.summary}")

    # ------------------- 结构化提取存储 -------------------
    def extract_structured_info(self, message: str):
        """
        从用户消息中提取结构化字段（姓名、偏好、职业等）。
        真实场景用 NER 或 LLM function calling。
        """
        extracted = {}
        # 模拟姓名提取
        name_match = re.search(r"我叫([\u4e00-\u9fa5]{2,4})", message)
        if name_match:
            extracted["姓名"] = name_match.group(1)

        # 模拟偏好提取（关键词匹配）
        if "喜欢" in message:
            if "Python" in message:
                extracted["偏好语言"] = "Python"
            if "咖啡" in message:
                extracted["偏好饮品"] = "咖啡"

        # 模拟职业提取
        if "程序员" in message or "开发" in message:
            extracted["职业"] = "软件工程师"

        if extracted:
            self.profile.update(extracted)
            self.pending_changes = True
            print(f"[结构化提取] 新提取字段: {extracted}")

    # ------------------- 触发写入机制 -------------------
    def on_user_message(self, content: str):
        """
        处理用户消息的入口，演示不同写入触发时机：
        - 每条消息自动触发结构化提取
        - 每 N 轮自动触发摘要更新
        - 也可手动调用强制写入
        """
        # 1. 原始存储（每条都存）
        self.add_message("user", content)

        # 2. 结构化提取（实时触发）
        self.extract_structured_info(content)

        # 3. 摘要压缩（按轮数触发）
        user_msg_count = sum(1 for m in self.raw_messages if "user:" in m)
        if user_msg_count % self.auto_summary_rounds == 0:
            self.generate_summary()

    def on_assistant_message(self, content: str):
        """助手消息也存入原始存储，但不做结构化提取"""
        self.add_message("assistant", content)

    def on_session_end(self):
        """
        会话结束时自动同步（触发写入的另一种时机）
        例如：强制生成最终摘要、持久化画像等
        """
        print("\n[会话结束] 触发自动同步...")
        if self.pending_changes:
            # 模拟持久化结构化信息
            print(f"[结构化存储] 最终画像: {json.dumps(self.profile, ensure_ascii=False)}")
            self.pending_changes = False
        self.generate_summary(force=True)

    def manual_write_instruction(self):
        """
        用户明确指令触发写入（例如用户说“记住这些信息”）
        """
        print("\n[手动触发] 用户明确指令要求写入记忆...")
        self.generate_summary(force=True)
        if self.profile:
            print(f"[结构化存储] 当前画像已确认: {json.dumps(self.profile, ensure_ascii=False)}")

    # ------------------- 查看状态 -------------------
    def show_status(self):
        print("\n====== 当前记忆状态 ======")
        print(f"原始消息数: {len(self.raw_messages)}")
        print(f"摘要内容: {self.summary or '无'}")
        print(f"结构化画像: {self.profile}")
        print("==========================\n")


# ========== 示例运行 ==========
if __name__ == "__main__":
    memory = MemoryManager()

    # 模拟多轮对话
    conversation = [
        ("user", "你好，我叫王小明，我是一名 Python 程序员。"),
        ("assistant", "你好王小明！有什么可以帮你的？"),
        ("user", "我想爬取豆瓣电影 Top250，用 requests 和 BeautifulSoup。"),
        ("assistant", "好的，requests + BeautifulSoup 是经典组合。遇到反爬问题了吗？"),
        ("user", "嗯，请求头反爬已经解决了，但是我还是很喜欢喝咖啡。"),
        ("assistant", "明白了，记得加上 User-Agent 和 Referer。"),
        ("user", "好的，先记住这些信息吧。"),          # 用户明确指令
        ("assistant", "已记录。还有什么需要？"),
        ("user", "暂时没了，今天就到这里。"),
        ("assistant", "再见！"),
    ]

    for role, text in conversation:
        if role == "user":
            memory.on_user_message(text)
            # 模拟用户明确指令触发
            if "记住这些信息" in text:
                memory.manual_write_instruction()
        else:
            memory.on_assistant_message(text)

    # 会话结束触发写入
    memory.on_session_end()

    # 展示最终状态
memory.show_status()