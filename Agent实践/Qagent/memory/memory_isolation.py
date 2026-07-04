from typing import Dict, List, Optional, Any
from collections import defaultdict
import json


class MemoryIsolationManager:
    """
    记忆隔离管理器，演示：
    1. 多租户隔离：按 tenant_id 划分独立命名空间
    2. 多角色/多任务隔离：在租户内部按 role_id 进一步隔离
    """

    def __init__(self):
        # 存储结构：tenant_id -> role_id -> memory_store
        # 其中 memory_store 可以是任何记忆存储后端（内存、向量库等）
        self._stores: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        # 用于权限校验的简单访问控制列表（可选）
        self._access_log: List[str] = []

    def _get_store(self, tenant_id: str, role_id: str) -> Dict[str, Any]:
        """获取指定租户和角色的存储字典（自动创建）"""
        return self._stores[tenant_id][role_id]

    def _log_access(self, tenant_id: str, role_id: str, action: str, key: str = ""):
        """记录访问日志（用于审计）"""
        log_entry = f"[{tenant_id}/{role_id}] {action} {key}"
        self._access_log.append(log_entry)
        # 实际应用中可写入日志文件或监控系统
        print(f"[审计] {log_entry}")

    # ---------- 多租户 + 多角色隔离的 CRUD ----------
    def add_memory(self, tenant_id: str, role_id: str, key: str, value: Any):
        """
        向指定租户的指定角色添加记忆。
        不同租户、不同角色之间的记忆完全隔离。
        """
        store = self._get_store(tenant_id, role_id)
        store[key] = value
        self._log_access(tenant_id, role_id, "ADD", key)

    def get_memory(self, tenant_id: str, role_id: str, key: str) -> Optional[Any]:
        """获取记忆，严格限定在 tenant_id + role_id 范围内"""
        store = self._get_store(tenant_id, role_id)
        value = store.get(key)
        self._log_access(tenant_id, role_id, "GET", key)
        return value

    def list_memories(self, tenant_id: str, role_id: str) -> Dict[str, Any]:
        """列出某租户某角色的所有记忆"""
        store = self._get_store(tenant_id, role_id)
        self._log_access(tenant_id, role_id, "LIST")
        return store.copy()

    def delete_memory(self, tenant_id: str, role_id: str, key: str):
        """删除记忆"""
        store = self._get_store(tenant_id, role_id)
        if key in store:
            del store[key]
            self._log_access(tenant_id, role_id, "DELETE", key)

    # ---------- 批量操作（租户或角色级别的清理）----------
    def clear_role_memories(self, tenant_id: str, role_id: str):
        """清空某租户下某角色的全部记忆"""
        if tenant_id in self._stores and role_id in self._stores[tenant_id]:
            self._stores[tenant_id][role_id].clear()
            self._log_access(tenant_id, role_id, "CLEAR_ROLE")

    def clear_tenant_memories(self, tenant_id: str):
        """清空某租户的所有记忆（所有角色）"""
        if tenant_id in self._stores:
            del self._stores[tenant_id]
            self._log_access(tenant_id, "*", "CLEAR_TENANT")

    # ---------- 隔离性验证辅助函数 ----------
    def show_all_stores(self):
        """打印当前所有存储的概览（仅用于调试，生产环境应禁用）"""
        print("\n====== 当前存储快照 ======")
        for tenant_id, roles in self._stores.items():
            print(f"租户: {tenant_id}")
            for role_id, store in roles.items():
                print(f"  角色: {role_id} -> {store}")
        print("==========================\n")


# ========== 演示运行 ==========
if __name__ == "__main__":
    mgr = MemoryIsolationManager()

    print("=== 场景：企业客服 Agent（多租户隔离）===")
    # 客户 A 的记忆
    mgr.add_memory("customer_a", "default", "name", "张三")
    mgr.add_memory("customer_a", "default", "issue", "无法登录账号")
    mgr.add_memory("customer_a", "default", "preferred_language", "中文")

    # 客户 B 的记忆
    mgr.add_memory("customer_b", "default", "name", "John Doe")
    mgr.add_memory("customer_b", "default", "issue", "Payment declined")
    mgr.add_memory("customer_b", "default", "preferred_language", "English")

    # 分别查看两个客户的记忆（隔离验证）
    print("\n客户 A 的记忆:")
    print(mgr.list_memories("customer_a", "default"))
    print("\n客户 B 的记忆:")
    print(mgr.list_memories("customer_b", "default"))

    # 尝试跨租户访问（将失败，因为 key 不存在于客户 B 中）
    cross_access = mgr.get_memory("customer_b", "default", "name")
    print(f"\n客户 B 中获取 'name' 得到: {cross_access}")  # John Doe，而不是张三

    print("\n=== 场景：同一 Agent 多角色隔离（客服 vs 技术支持）===")
    # 假设租户 "enterprise_001" 内部有客服角色和技术支持角色
    mgr.add_memory("enterprise_001", "customer_service", "greeting", "您好，请问有什么可以帮您？")
    mgr.add_memory("enterprise_001", "customer_service", "escalation_policy", "超过30分钟未解决转接技术支持")

    mgr.add_memory("enterprise_001", "tech_support", "knowledge_base", "常见错误码 500 解决方案...")
    mgr.add_memory("enterprise_001", "tech_support", "tools", "可执行远程诊断命令")

    print("\n客服角色的记忆:")
    print(mgr.list_memories("enterprise_001", "customer_service"))
    print("\n技术支持角色的记忆:")
    print(mgr.list_memories("enterprise_001", "tech_support"))

    # 验证隔离：客服无法获取技术支持的知识库
    tech_kb = mgr.get_memory("enterprise_001", "customer_service", "knowledge_base")
    print(f"\n客服角色获取 'knowledge_base': {tech_kb}")  # None

    print("\n=== 完整存储结构 ===")
    mgr.show_all_stores()

    print("=== 清理操作演示 ===")
    mgr.clear_role_memories("enterprise_001", "customer_service")
    print("清空企业001的客服角色记忆后:")
    print(mgr.list_memories("enterprise_001", "customer_service"))  # 空字典

    mgr.clear_tenant_memories("customer_a")
    print("清空客户A的全部记忆后，租户customer_a不存在于存储中:")
print("customer_a" in mgr._stores)  # False