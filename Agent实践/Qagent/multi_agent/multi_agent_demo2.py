#!/usr/bin/env python3
"""
多智能体协作演示 Demo
基于文档中的核心概念：工厂模式、注册表、星型拓扑、上下文管理、生命周期管理
"""

import uuid
import json
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed


# ==================== 1. 消息协议 ====================

@dataclass
class Message:
    """Agent 间通信的标准消息信封"""
    from_agent: str
    to_agent: str
    content: Any
    msg_type: str = "task"  # task | result | inform | error
    context_ref: Dict = field(default_factory=dict)
    trace_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.msg_type,
            "content": self.content,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp
        }


# ==================== 2. 上下文管理（集中式共享） ====================

class TaskContext:
    """全局任务上下文 - 由 Orchestrator 持有，Agent 不直接访问"""

    def __init__(self, task_id: str, goal: str):
        self.task_id = task_id
        self.goal = goal
        self.sections: Dict[str, Any] = {}
        self.status = "running"
        self.created_at = time.time()
        self.logs: List[Dict] = []

    def write(self, section: str, data: Any, author: str):
        self.sections[section] = {
            "data": data,
            "author": author,
            "time": time.time()
        }
        self.logs.append({"action": "write", "section": section, "author": author})

    def read(self, section: str) -> Optional[Any]:
        return self.sections.get(section, {}).get("data")

    def summary(self) -> str:
        return json.dumps({
            "task_id": self.task_id,
            "goal": self.goal,
            "sections": list(self.sections.keys()),
            "status": self.status
        }, indent=2, ensure_ascii=False)


class ContextStore:
    """上下文存储服务 - 集中式管理所有任务上下文"""

    def __init__(self):
        self._store: Dict[str, TaskContext] = {}

    def create(self, goal: str) -> TaskContext:
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        ctx = TaskContext(task_id, goal)
        self._store[task_id] = ctx
        return ctx

    def get(self, task_id: str) -> Optional[TaskContext]:
        return self._store.get(task_id)

    def remove(self, task_id: str):
        if task_id in self._store:
            del self._store[task_id]


# ==================== 3. Agent 基类与注册表 ====================

class BaseAgent:
    """Agent 基类 - 只持有私有上下文，不持有全局上下文引用"""

    def __init__(self, agent_id: str, role: str, orchestrator: 'Orchestrator'):
        self.agent_id = agent_id
        self.role = role
        self.orchestrator = orchestrator  # 仅用于发消息，不直接访问全局上下文
        self.private_context: Dict = {}   # 私有推理暂存区
        self.created_at = time.time()

    def handle_message(self, msg: Message) -> Message:
        """核心处理逻辑 - 子类必须实现"""
        raise NotImplementedError

    def _build_briefing(self, task: str, snapshot: Dict) -> str:
        """构建任务简报 - 模拟从消息中提取上下文"""
        return f"""[角色: {self.role}] [ID: {self.agent_id}]
任务: {task}
相关背景: {json.dumps(snapshot, ensure_ascii=False)}
请完成上述任务并返回结构化结果。"""

    def destroy(self):
        """清理资源"""
        self.private_context.clear()
        print(f"  🗑️ Agent {self.agent_id} 已销毁")


# ==================== 4. 具体 Agent 实现 ====================

class AnalyzerAgent(BaseAgent):
    """分析型 Agent - 负责需求分析和方案设计"""

    def handle_message(self, msg: Message) -> Message:
        task = msg.content.get("task", "")
        snapshot = msg.context_ref.get("snapshot", {})

        # 模拟分析工作
        print(f"  🔍 [{self.agent_id}] 正在分析任务: {task[:30]}...")
        time.sleep(0.5)  # 模拟耗时

        # 模拟 LLM 推理过程（私有上下文）
        self.private_context["reasoning"] = f"分析步骤: 1)理解需求 2)拆解模块 3)评估风险"

        result = {
            "analysis": {
                "modules": ["auth", "api", "db"],
                "risks": ["并发安全", "数据一致性"],
                "approach": "微服务架构"
            },
            "confidence": 0.92
        }

        return Message(
            from_agent=self.agent_id,
            to_agent=msg.from_agent,  # 返回给发送者（Orchestrator）
            content=result,
            msg_type="result",
            trace_id=msg.trace_id
        )


class CoderAgent(BaseAgent):
    """编码型 Agent - 负责代码实现"""

    def handle_message(self, msg: Message) -> Message:
        task = msg.content.get("task", "")
        analysis = msg.context_ref.get("analysis", {})

        print(f"  💻 [{self.agent_id}] 正在编码: {task[:30]}...")
        time.sleep(0.6)

        modules = analysis.get("modules", ["core"])
        code = {m: f"# {m} module implementation\nclass {m.title()}Service: pass" 
                for m in modules}

        return Message(
            from_agent=self.agent_id,
            to_agent=msg.from_agent,
            content={"code": code, "files": list(code.keys())},
            msg_type="result",
            trace_id=msg.trace_id
        )


class ReviewerAgent(BaseAgent):
    """审查型 Agent - 负责代码审查"""

    def handle_message(self, msg: Message) -> Message:
        code = msg.content.get("code", {})

        print(f"  👀 [{self.agent_id}] 正在审查代码...")
        time.sleep(0.4)

        issues = []
        for file in code:
            if "auth" in file.lower():
                issues.append({"file": file, "severity": "high", "issue": "缺少输入校验"})

        return Message(
            from_agent=self.agent_id,
            to_agent=msg.from_agent,
            content={
                "passed": len(issues) == 0,
                "issues": issues,
                "suggestions": ["添加参数校验", "增加日志记录"] if issues else []
            },
            msg_type="result",
            trace_id=msg.trace_id
        )


class TesterAgent(BaseAgent):
    """测试型 Agent - 负责生成测试用例"""

    def handle_message(self, msg: Message) -> Message:
        code = msg.content.get("code", {})

        print(f"  🧪 [{self.agent_id}] 正在生成测试...")
        time.sleep(0.4)

        tests = {f"test_{k}.py": f"def test_{k}(): assert True  # TODO" 
                 for k in code.keys()}

        return Message(
            from_agent=self.agent_id,
            to_agent=msg.from_agent,
            content={"tests": tests, "coverage": 0.85},
            msg_type="result",
            trace_id=msg.trace_id
        )


# ==================== 5. 注册表 + 工厂 ====================

class AgentRegistry:
    """Agent 注册表 - 运行时动态扩展角色映射"""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, role: str, agent_cls: type):
        cls._registry[role] = agent_cls
        print(f"  📋 注册角色: {role} -> {agent_cls.__name__}")

    @classmethod
    def create(cls, role: str, agent_id: str, orchestrator: 'Orchestrator', 
               briefing: Dict = None) -> BaseAgent:
        agent_cls = cls._registry.get(role)
        if not agent_cls:
            raise ValueError(f"未知角色: {role}，可用: {list(cls._registry.keys())}")

        agent = agent_cls(agent_id, role, orchestrator)
        if briefing:
            agent.private_context["briefing"] = briefing
        return agent

    @classmethod
    def list_roles(cls) -> List[str]:
        return list(cls._registry.keys())


# ==================== 6. 编排器（星型拓扑中心） ====================

class Orchestrator:
    """中心编排器 - 星型拓扑的 Hub，负责创建、调度、销毁 Agent"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}      # 活跃 Agent 注册表
        self.context_store = ContextStore()          # 全局上下文服务
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.stats = {"created": 0, "destroyed": 0}

    # ---- 生命周期管理 ----

    def create_agent(self, role: str, briefing: Dict = None) -> str:
        """工厂方法：动态创建 Agent"""
        agent_id = f"{role}-{uuid.uuid4().hex[:6]}"
        agent = AgentRegistry.create(role, agent_id, self, briefing)
        self.agents[agent_id] = agent
        self.stats["created"] += 1
        print(f"  ✅ 创建 Agent: {agent_id} (角色: {role})")
        return agent_id

    def destroy_agent(self, agent_id: str):
        """销毁 Agent - 从注册表移除并清理资源"""
        if agent_id in self.agents:
            self.agents[agent_id].destroy()
            del self.agents[agent_id]
            self.stats["destroyed"] += 1

    def dispatch(self, agent_id: str, msg: Message, timeout: float = 10.0) -> Message:
        """向指定 Agent 发送消息并等待返回（带超时）"""
        if agent_id not in self.agents:
            return Message("system", "orchestrator", 
                        {"error": f"Agent {agent_id} 不存在"}, "error")

        agent = self.agents[agent_id]
        print(f"  📤 Orchestrator -> {agent_id}: {msg.msg_type}")

        # 模拟异步执行（实际可用 asyncio）
        future = self.executor.submit(agent.handle_message, msg)
        try:
            result = future.result(timeout=timeout)
            print(f"  📥 {agent_id} -> Orchestrator: {result.msg_type}")
            return result
        except Exception as e:
            print(f"  ⚠️ Agent {agent_id} 执行异常: {e}")
            return Message(agent_id, "orchestrator", {"error": str(e)}, "error")

    def broadcast(self, agent_ids: List[str], msg: Message) -> List[Message]:
        """并行分派给多个 Agent"""
        futures = {aid: self.executor.submit(self.dispatch, aid, msg) 
                   for aid in agent_ids}
        results = []
        for aid, future in futures.items():
            try:
                results.append(future.result(timeout=15))
            except Exception as e:
                results.append(Message(aid, "orchestrator", {"error": str(e)}, "error"))
        return results

    # ---- 工作流编排 ----

    def run_pipeline(self, goal: str) -> Dict:
        """执行完整的多 Agent 协作流水线"""
        print("\n" + "="*60)
        print(f"🚀 启动任务: {goal}")
        print("="*60)

        # 1. 创建任务上下文
        ctx = self.context_store.create(goal)

        # 2. 阶段一：分析（串行）
        print("\n📌 阶段 1: 需求分析")
        analyzer_id = self.create_agent("analyzer")
        analysis_msg = Message(
            from_agent="orchestrator",
            to_agent=analyzer_id,
            content={"task": goal},
            context_ref={"snapshot": {"goal": goal, "constraints": ["高性能", "可扩展"]}},
            trace_id=ctx.task_id
        )
        analysis_result = self.dispatch(analyzer_id, analysis_msg)
        ctx.write("analysis", analysis_result.content, analyzer_id)
        self.destroy_agent(analyzer_id)  # 任务完成即销毁

        # 3. 阶段二：编码 + 测试（并行）
        print("\n📌 阶段 2: 并行编码与测试准备")
        coder_id = self.create_agent("coder")
        tester_id = self.create_agent("tester")

        code_msg = Message(
            from_agent="orchestrator",
            to_agent=coder_id,
            content={"task": goal},
            context_ref={"analysis": analysis_result.content.get("analysis", {})},
            trace_id=ctx.task_id
        )
        test_msg = Message(
            from_agent="orchestrator",
            to_agent=tester_id,
            content={"task": goal},
            context_ref={"analysis": analysis_result.content.get("analysis", {})},
            trace_id=ctx.task_id
        )

        # 并行执行
        parallel_results = self.broadcast([coder_id, tester_id], 
                                         Message("orchestrator", "parallel", 
                                                {"task": goal}, "task",
                                                {"analysis": analysis_result.content.get("analysis", {})},
                                                ctx.task_id))
        # 修正：分别发送不同消息
        code_result = self.dispatch(coder_id, code_msg)
        test_result = self.dispatch(tester_id, test_msg)

        ctx.write("code", code_result.content, coder_id)
        ctx.write("tests", test_result.content, tester_id)

        self.destroy_agent(coder_id)
        self.destroy_agent(tester_id)

        # 4. 阶段三：审查（串行，依赖编码结果）
        print("\n📌 阶段 3: 代码审查")
        reviewer_id = self.create_agent("reviewer")
        review_msg = Message(
            from_agent="orchestrator",
            to_agent=reviewer_id,
            content={"code": code_result.content.get("code", {})},
            trace_id=ctx.task_id
        )
        review_result = self.dispatch(reviewer_id, review_msg)
        ctx.write("review", review_result.content, reviewer_id)
        self.destroy_agent(reviewer_id)

        # 5. 汇总
        print("\n📌 阶段 4: 结果汇总")
        ctx.status = "completed"

        final_report = {
            "task_id": ctx.task_id,
            "goal": goal,
            "analysis": ctx.read("analysis"),
            "code_files": ctx.read("code"),
            "tests": ctx.read("tests"),
            "review": ctx.read("review"),
            "stats": self.stats.copy()
        }

        print("\n" + "="*60)
        print("📊 任务完成报告")
        print("="*60)
        print(json.dumps(final_report, indent=2, ensure_ascii=False))

        return final_report

    def shutdown(self):
        """优雅关闭 - 清理所有资源"""
        print("\n🛑 优雅关闭中...")
        for aid in list(self.agents.keys()):
            self.destroy_agent(aid)
        self.executor.shutdown(wait=True)
        print(f"📈 统计: 创建 {self.stats['created']} 个, 销毁 {self.stats['destroyed']} 个")


# ==================== 7. 动态扩展演示 ====================

class CustomAgent(BaseAgent):
    """自定义 Agent - 演示运行时动态注册新角色"""

    def handle_message(self, msg: Message) -> Message:
        print(f"  🎨 [{self.agent_id}] 自定义 Agent 处理中...")
        return Message(
            from_agent=self.agent_id,
            to_agent=msg.from_agent,
            content={"custom_output": "这是自定义 Agent 的输出", "input_echo": msg.content},
            msg_type="result",
            trace_id=msg.trace_id
        )


# ==================== 8. 主程序 ====================

def main():
    print("="*60)
    print("🤖 多智能体协作系统 Demo")
    print("="*60)

    # 1. 初始化注册表（预定义角色）
    print("\n🔧 初始化 Agent 注册表...")
    AgentRegistry.register("analyzer", AnalyzerAgent)
    AgentRegistry.register("coder", CoderAgent)
    AgentRegistry.register("reviewer", ReviewerAgent)
    AgentRegistry.register("tester", TesterAgent)

    # 2. 运行时动态扩展：注册自定义 Agent
    print("\n🔧 动态扩展：注册自定义 Agent 角色 'custom'")
    AgentRegistry.register("custom", CustomAgent)

    # 3. 创建编排器
    orch = Orchestrator()

    # 4. 执行主任务流水线
    result = orch.run_pipeline("设计并实现一个用户认证微服务，支持JWT和OAuth2")

    # 5. 演示：销毁后重新创建同名角色 Agent
    print("\n📌 演示: 销毁后重新创建同角色 Agent")
    aid1 = orch.create_agent("analyzer", {"task": "二次分析"})
    orch.destroy_agent(aid1)
    aid2 = orch.create_agent("analyzer", {"task": "三次分析"})  # 全新实例
    orch.destroy_agent(aid2)

    # 6. 演示：动态创建的 custom Agent
    print("\n📌 演示: 调用动态扩展的 Custom Agent")
    custom_id = orch.create_agent("custom")
    resp = orch.dispatch(custom_id, Message(
        "orchestrator", custom_id, {"hello": "world"}, "task", {}, result["task_id"]
    ))
    orch.destroy_agent(custom_id)

    # 7. 优雅关闭
    orch.shutdown()

    print("\n✅ Demo 执行完毕！")


if __name__ == "__main__":
    main()
