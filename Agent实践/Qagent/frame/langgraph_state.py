import uuid
import operator
from typing import TypedDict, List, Annotated

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    # 关键：使用 operator.add 作为 reducer，实现消息追加而非覆盖
    messages: Annotated[List[str], operator.add]


def node_a(state: State):
    print("执行节点 A")
    return {"messages": ["A 说：处理了第一步"]}


def node_b(state: State):
    print("执行节点 B")
    return {"messages": ["B 说：处理了第二步"]}


# 构建图
builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.set_entry_point("node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 同一个线程 ID
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# ---------- 第一次调用 ----------
print("=== 第一次调用 ===")
result1 = graph.invoke({"messages": []}, config)
print("最终 state:", result1)
print("消息列表:", result1["messages"])
# 输出：['A 说：处理了第一步', 'B 说：处理了第二步']

# ---------- 第二次调用（继续同线程）----------
print("\n=== 第二次调用（继续同一会话） ===")
# 不再传入空的 messages 列表，而是传 None 或者不传（只传配置），
# 但即使传入空列表，因为有 operator.add，空列表会与旧列表拼接，不会覆盖。
result2 = graph.invoke({"messages": []}, config)
print("最终 state:", result2)
print("消息列表:", result2["messages"])
# 输出：['A...第一步', 'B...第二步', 'A...第一步', 'B...第二步']
# 历史累积！

# ---------- 第三次调用（新线程）----------
new_thread_id = str(uuid.uuid4())
new_config = {"configurable": {"thread_id": new_thread_id}}
print("\n=== 新线程调用 ===")
result3 = graph.invoke({"messages": []}, new_config)
print("最终 state:", result3)
print("消息列表:", result3["messages"])
# 输出：['A...第一步', 'B...第二步']   ← 新对话从头开始