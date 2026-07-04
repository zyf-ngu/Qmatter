import operator
from typing import TypedDict, List, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command, Send
from langgraph.config import get_stream_writer

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# ---------- 1. 状态定义 ----------
# messages 字段使用 operator.add 作为 reducer，自动追加新消息
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # Reducer：消息列表追加
    intent: str
    action_approved: bool


# ---------- 2. 节点函数 ----------
def intent_classifier(state: AgentState) -> dict:
    """模拟意图分类：简单判断用户输入中是否包含‘工具’关键词"""
    last_msg = state["messages"][-1].content.lower()
    if "工具" in last_msg:
        return {"intent": "use_tool"}
    return {"intent": "chat"}


def tool_executor(state: AgentState) -> dict:
    """执行工具（需要人工审批）"""
    # 中断并等待人工审批
    approval = interrupt("请批准执行工具操作吗？(yes/no)")
    if approval.lower() != "yes":
        return {"action_approved": False, "messages": [AIMessage(content="工具执行已被拒绝。")]}

    # 模拟工具执行，并通过流式写入器发送进度
    writer = get_stream_writer()
    writer("开始执行工具...")
    writer("进度: 50%")
    writer("进度: 100%，工具执行完毕。")

    return {"action_approved": True, "messages": [AIMessage(content="工具执行成功：数据已处理。")]}


def chat_responder(state: AgentState) -> dict:
    """普通聊天回复"""
    return {"messages": [AIMessage(content="你好！我是一个简单的助手。")]}


def should_continue(state: AgentState) -> str:
    """条件路由：根据意图决定下一步"""
    if state["intent"] == "use_tool":
        return "tool_node"
    return "chat_node"


# ---------- 3. 构建图 ----------
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("classifier", intent_classifier)
workflow.add_node("tool_node", tool_executor)
workflow.add_node("chat_node", chat_responder)

# 添加边
workflow.add_edge(START, "classifier")
workflow.add_conditional_edges("classifier", should_continue, {
    "tool_node": "tool_node",
    "chat_node": "chat_node"
})
workflow.add_edge("tool_node", END)
workflow.add_edge("chat_node", END)

# ---------- 4. 编译并启用持久化 ----------
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ---------- 5. 演示运行 ----------
if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "demo-session"}}

    # 第一轮：触发工具调用（中断并等待人工审批）
    print("=== 第一轮调用 ===")
    user_input = {"messages": [HumanMessage(content="请帮我使用工具处理数据")]}
    try:
        # stream_mode="updates" 可观察每一步的状态增量
        for event in app.stream(user_input, thread, stream_mode="updates"):
            print("事件:", event)
    except Exception as e:
        # 中断时会抛出异常，捕获后可以进行人工输入
        pass

    # 此时图已中断，需要人工恢复
    print("\n>>> 人工审批：输入 yes 或 no")
    human_decision = input("批准操作？(yes/no): ")

    # 恢复执行，Command(resume=...) 传递中断处的返回值
    for event in app.stream(Command(resume=human_decision), thread, stream_mode="updates"):
        print("恢复后事件:", event)

    # 第二轮：普通对话（无中断）
    print("\n=== 第二轮调用 ===")
    user_input = {"messages": [HumanMessage(content="你好")]}
    for event in app.stream(user_input, thread, stream_mode="updates"):
        print("事件:", event)

    # 流式输出消息内容（可选演示 stream_mode="messages"）
    print("\n=== 流式输出消息 ===")
    user_input = {"messages": [HumanMessage(content="请再次使用工具")]}
    # 同样会中断，此处仅演示流式输出的方式（通常与支持流式的LLM结合）
    # for chunk in app.stream(user_input, thread, stream_mode="messages"):
    #     print(chunk, end="|")