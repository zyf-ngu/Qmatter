import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

# ---------- 消息结构 ----------
@dataclass
class Message:
    from_agent: str
    to_agent: str
    content: Dict[str, Any]
    context_ref: Dict[str, Any] = field(default_factory=dict)  # 从Orchestrator传递的上下文快照

# ---------- 基础Agent ----------
class BaseAgent:
    def __init__(self, agent_id: str, role: str, orchestrator: "Orchestrator"):
        self.agent_id = agent_id
        self.role = role
        self.orchestrator = orchestrator
        self.private_context: Dict[str, Any] = {}  # 私有暂存，仅内部使用

    def handle_message(self, msg: Message) -> Dict[str, Any]:
        """处理消息，返回结果（可包含上下文更新建议）"""
        raise NotImplementedError

# ---------- 具体子Agent ----------
class AnalyzerAgent(BaseAgent):
    def handle_message(self, msg: Message) -> Dict[str, Any]:
        print(f"[{self.agent_id}] 分析任务: {msg.content.get('goal')}")
        # 模拟分析过程
        analysis = f"需求分析完成：需要实现加减乘除功能"
        self.private_context["last_analysis"] = analysis
        return {
            "status": "success",
            "result": analysis,
            "context_update": {"analysis": analysis}  # 建议更新全局上下文
        }

class CoderAgent(BaseAgent):
    def handle_message(self, msg: Message) -> Dict[str, Any]:
        print(f"[{self.agent_id}] 编码任务: {msg.content.get('goal')}")
        # 模拟代码生成
        code = "def add(a,b): return a+b\n" + "def sub(a,b): return a-b\n"
        self.private_context["last_code"] = code
        return {
            "status": "success",
            "result": code,
            "context_update": {"code": code}
        }

# ---------- Agent注册表（支持动态扩展） ----------
class AgentRegistry:
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, role: str, agent_cls: type):
        cls._registry[role] = agent_cls
        print(f"[Registry] 注册角色: {role} -> {agent_cls.__name__}")

    @classmethod
    def create(cls, role: str, agent_id: str, orchestrator: "Orchestrator") -> BaseAgent:
        agent_cls = cls._registry.get(role)
        if not agent_cls:
            raise ValueError(f"未知角色: {role}")
        return agent_cls(agent_id, role, orchestrator)

# ---------- 编排器（Orchestrator） ----------
class Orchestrator:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}          # 动态子Agent实例
        self.global_context: Dict[str, Any] = {         # 全局共享上下文（集中式）
            "task": "",
            "analysis": "",
            "code": ""
        }
        self.trace_id = str(uuid.uuid4())[:8]

    def plan(self, user_task: str) -> List[Dict[str, str]]:
        """规划子任务（演示硬编码分解）"""
        print(f"[Orchestrator] 规划任务: {user_task}")
        # 简单规则：分析 + 编码
        return [
            {"role": "analyzer", "goal": "分析需求"},
            {"role": "coder", "goal": "编写代码"}
        ]

    def decompose_and_execute(self, user_task: str):
        """分解并执行任务"""
        self.global_context["task"] = user_task

        # 1. 规划
        sub_tasks = self.plan(user_task)

        # 2. 为每个子任务动态创建Agent并执行
        results = []
        for task in sub_tasks:
            role = task["role"]
            agent_id = f"{role}-{uuid.uuid4().hex[:6]}"

            # 动态创建（通过注册表）
            agent = AgentRegistry.create(role, agent_id, self)
            self.agents[agent_id] = agent
            print(f"[Orchestrator] 创建 Agent: {agent_id}")

            # 准备消息（携带必要的上下文快照）
            context_snapshot = {
                "task_summary": self.global_context.get("task"),
                "prev_results": {k: v for k, v in self.global_context.items() if k not in ["task"]}
            }
            msg = Message(
                from_agent="orchestrator",
                to_agent=agent_id,
                content={"goal": task["goal"]},
                context_ref=context_snapshot
            )

            # 执行
            response = agent.handle_message(msg)
            results.append(response)

            # 3. 合并上下文更新
            if "context_update" in response:
                for key, value in response["context_update"].items():
                    if key in self.global_context:
                        self.global_context[key] = value
                        print(f"[Orchestrator] 更新全局上下文: {key} = {value[:30]}...")

            # 4. 销毁子Agent（任务完成即回收）
            self.destroy_agent(agent_id)

        # 5. 聚合最终结果
        final = self.aggregate(results)
        print(f"[Orchestrator] 最终结果: {final}")
        return final

    def destroy_agent(self, agent_id: str):
        """从注册表中移除并清理资源"""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            # 可选：清空引用，帮助GC
            agent.orchestrator = None
            print(f"[Orchestrator] 销毁 Agent: {agent_id}")

    def aggregate(self, results: List[Dict]) -> str:
        """汇总子Agent结果"""
        analysis = self.global_context.get("analysis", "")
        code = self.global_context.get("code", "")
        return f"任务完成！分析：{analysis}\n代码：\n{code}"

# ---------- 使用示例 ----------
def main():
    # 1. 注册Agent类（预定义，支持动态扩展）
    AgentRegistry.register("analyzer", AnalyzerAgent)
    AgentRegistry.register("coder", CoderAgent)

    # 2. 创建编排器
    orchestrator = Orchestrator()

    # 3. 提交用户任务
    orchestrator.decompose_and_execute("开发一个简单的计算器功能")

if __name__ == "__main__":
    main()