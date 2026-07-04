from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---------- 1. 模拟 LLM ----------
class FakeLLM(LLM):
    """模拟大模型，打印输入并返回固定答案"""
    @property
    def _llm_type(self) -> str:
        return "fake"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        print("\n===== LLM 收到的 Prompt =====")
        print(prompt[:600])
        return "（模拟答案）核心观点：架构的核心是模块化与组合性。"

# ---------- 2. 模拟检索器 ----------
class FakeRetriever(BaseRetriever):
    """永远返回两条固定文档片段"""
    def _get_relevant_documents(self, query: str) -> List[Document]:
        print(f"\n>> 检索查询：{query}")
        return [
            Document(page_content="架构的核心思想是模块化与组合性，每个组件可独立开发和替换。"),
            Document(page_content="LLM 应用中，提示、模型、输出解析应设计为可插拔的管道。")
        ]

# ---------- 3. 构建 RAG 管道 ----------
# 3.1 定义提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "使用下面的上下文回答问题。不知道就说不知道。\n\n上下文：\n{context}"),
    ("human", "{input}")
])

# 3.2 将检索到的文档合并为纯文本
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# 3.3 组合管道（完全等价于 RetrievalQA + stuff）
rag_chain = (
    {
        "context": RunnableLambda(lambda x: x["input"]) | FakeRetriever() | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | FakeLLM()
    | StrOutputParser()
)

# ---------- 4. 测试 ----------



from typing import List, Optional
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun

# ---------- 1. 定义工具 ----------
@tool
def weather_query(city: str) -> str:
    """查询指定城市的天气，输入城市名称字符串。"""
    weather_db = {"上海": "晴，25°C", "北京": "多云，18°C"}
    return weather_db.get(city, f"未找到{city}的天气")

tools = [weather_query]

# ---------- 2. 模拟 ChatModel ----------
class FakeChatModel(BaseChatModel):
    """根据用户输入返回预设的 AIMessage，支持 tool_calls"""
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        user_input = messages[-1].content if messages else ""
        if "上海" in user_input:
            # 返回带工具调用的消息
            ai_msg = AIMessage(
                content="",
                tool_calls=[{"name": "weather_query", "args": {"city": "上海"}, "id": "call_001"}]
            )
        elif "笑话" in user_input:
            ai_msg = AIMessage(content="抱歉，我没有讲笑话的工具。")
        else:
            ai_msg = AIMessage(content="我无法理解你的请求。")
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

# ---------- 3. 创建 Agent ----------
agent = create_react_agent(FakeChatModel(), tools)

# ---------- 4. 测试 ----------
if __name__ == "__main__":
    question = "这篇文档的核心观点是什么？"
    answer = rag_chain.invoke({"input": question})
    print("\n===== 最终答案 =====")
    print(answer)


    print("===== 测试1：查询天气 =====")
    res = agent.invoke({"messages": [("user", "查询上海天气")]})
    print("最终输出：", res["messages"][-1].content)

    print("\n===== 测试2：讲笑话 =====")
    res = agent.invoke({"messages": [("user", "讲一个笑话")]})
    print("最终输出：", res["messages"][-1].content)