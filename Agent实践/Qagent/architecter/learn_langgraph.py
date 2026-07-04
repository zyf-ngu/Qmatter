"""
LangGraph 综合特性演示 Demo
包含：状态管理、循环控制、条件边、检查点持久化、流式输出、人机协作、
     工具调用、长期记忆、上下文窗口管理
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.errors import GraphInterrupt

# ================== 1. 状态定义 ==================
class AgentState(TypedDict):
    task: str
    draft: str
    feedback: str
    attempt: int
    final_answer: str
    # 新增字段
    messages: list[dict]          # 上下文窗口（保留最近 N 条消息）
    memory: str                   # 长期记忆（用户偏好等）
    tool_results: list[str]       # 工具调用历史（展示用）

# ================== 2. 工具定义 ==================
def search_tool(query: str) -> str:
    """模拟搜索工具，返回虚假数据"""
    return f"搜索结果：关于“{query}”的最新市场数据显示增长趋势为 12%。"

# ================== 3. 节点定义 ==================
def tool_node(state: AgentState) -> dict:
    """工具调用节点：根据任务和已有信息决定是否调用工具"""
    print("[工具节点] 检查是否需要调用工具...")
    # 简单规则：如果草稿为空（初次）或反馈要求补充数据，则调用搜索工具
    if not state["draft"] or "数据" in state["feedback"]:
        query = state["task"]
        tool_output = search_tool(query)
        print(f"[工具节点] 调用搜索工具，结果：{tool_output}")
        # 更新工具调用历史
        new_tools = state["tool_results"] + [tool_output]
        # 将工具调用记录到上下文消息中
        tool_msg = {"role": "tool", "content": tool_output}
        new_messages = state["messages"] + [tool_msg]
        return {"tool_results": new_tools, "messages": new_messages}
    return {}  # 不调用工具则不更新状态

def generate_draft(state: AgentState) -> dict:
    """生成草稿：结合记忆、工具结果、上下文窗口"""
    print("[生成草稿] 综合信息...")
    # 模拟：从长期记忆中提取用户偏好
    preference = state["memory"] if state["memory"] else "无特殊偏好"
    # 准备上下文窗口（最近 3 条消息）
    recent_context = state["messages"][-3:] if state["messages"] else []
    context_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent_context])

    # 生成草稿（模拟，实际会调用 LLM）
    if state["draft"]:
        # 有草稿时根据反馈修改
        draft = f"（修改版）基于反馈“{state['feedback']}”和工具数据，优化后的方案"
    else:
        draft = f"初始方案：为任务“{state['task']}”生成计划。用户偏好：{preference}"
    if context_str:
        draft += f"\n参考上下文：{context_str}"

    # 更新上下文消息
    ai_msg = {"role": "ai", "content": draft}
    new_messages = state["messages"] + [ai_msg]
    # 上下文窗口控制：只保留最近 10 条
    if len(new_messages) > 10:
        new_messages = new_messages[-10:]

    return {
        "draft": draft,
        "attempt": state["attempt"] + 1,
        "messages": new_messages
    }

def evaluate_draft(state: AgentState) -> dict:
    """自动评估草稿"""
    print(f"[自动评估] 当前草稿: {state['draft'][:50]}...")
    # 模拟反馈：前两次尝试要求补充数据，之后通过
    if state["attempt"] < 3:
        auto_feedback = "内容需补充更多数据"
    else:
        auto_feedback = "基本合格"
    return {"feedback": auto_feedback}

def human_review(state: AgentState) -> dict:
    """人机协作节点：暂停并等待人类审核，同时处理记忆和上下文"""
    print("\n🛑 暂停，等待人工审核...")
    prompt = (
        f"当前草稿:\n{state['draft']}\n\n"
        f"自动反馈: {state['feedback']}\n"
        "你可以输入：\n"
        "  - '通过' 结束任务\n"
        "  - 具体反馈进入下一轮优化\n"
        "  - '记忆:xxx' 来更新长期记忆（如：记忆:用户喜欢图表）\n"
        "请输入："
    )
    user_input = interrupt(prompt)

    # 处理记忆指令
    new_memory = state["memory"]
    if user_input.startswith("记忆:"):
        mem_content = user_input[3:].strip()
        new_memory = mem_content
        print(f"[记忆更新] 新记忆: {new_memory}")

    # 将人类输入加入上下文消息
    human_msg = {"role": "human", "content": user_input}
    new_messages = state["messages"] + [human_msg]
    if len(new_messages) > 10:
        new_messages = new_messages[-10:]

    return {
        "feedback": user_input,   # 用人类输入作为最终反馈
        "memory": new_memory,
        "messages": new_messages
    }

def finalize(state: AgentState) -> dict:
    """最终化"""
    print("[最终化] 任务完成")
    return {"final_answer": state["draft"]}

# ================== 4. 条件边 ==================
def should_continue(state: AgentState) -> Literal["refine", "finish"]:
    if state["feedback"] == "通过":
        return "finish"
    return "refine"

# ================== 5. 构建图 ==================
builder = StateGraph(AgentState)

builder.add_node("tool", tool_node)
builder.add_node("generate", generate_draft)
builder.add_node("evaluate", evaluate_draft)
builder.add_node("human_review", human_review)
builder.add_node("finalize", finalize)

# 流程：工具 -> 生成 -> 评估 -> 人类审核 -> 条件 -> 循环回工具或结束
builder.add_edge(START, "tool")
builder.add_edge("tool", "generate")
builder.add_edge("generate", "evaluate")
builder.add_edge("evaluate", "human_review")
builder.add_conditional_edges(
    "human_review",
    should_continue,
    {"refine": "tool", "finish": "finalize"}
)
builder.add_edge("finalize", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ================== 6. 运行演示 ==================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-full"}}
    initial_state = {
        "task": "市场分析报告",
        "attempt": 0,
        "draft": "",
        "feedback": "",
        "final_answer": "",
        "messages": [],
        "memory": "",
        "tool_results": []
    }

    print("======== 开始执行（将因中断暂停） ========")
    try:
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, update in event.items():
                print(f"✅ 节点 [{node_name}] 输出: {update}")
    except GraphInterrupt:
        print("⏸️ 已暂停，等待人工输入...\n")

    # 模拟人类交互（多次，直到任务完成）
    while True:
        state = graph.get_state(config)
        # 检查是否已完成
        if state.values.get("final_answer"):
            break

        print("\n--- 当前状态（供审核）---")
        print(f"草稿: {state.values['draft'][:100]}...")
        print(f"记忆: {state.values['memory']}")
        print("-------------------------")
        user_cmd = input("请输入指令（'通过'/反馈/'记忆:内容'）: ").strip()
        if not user_cmd:
            user_cmd = "通过"

        print("\n======== 恢复执行 ========")
        try:
            for event in graph.stream(Command(resume=user_cmd), config, stream_mode="updates"):
                for node_name, update in event.items():
                    print(f"✅ 节点 [{node_name}] 输出: {update}")
        except GraphInterrupt:
            print("⏸️ 再次暂停，继续等待输入...\n")
        else:
            break  # 如果没有中断，流程结束

    print("\n======== 最终结果 ========")
    final = graph.get_state(config)
    print("最终答案:", final.values.get("final_answer"))
    print("完整状态摘要:")
    print("  尝试次数:", final.values["attempt"])
    print("  上下文消息数:", len(final.values["messages"]))
    print("  长期记忆:", final.values["memory"])
    print("  工具调用记录:", final.values["tool_results"])