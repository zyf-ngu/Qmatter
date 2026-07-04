import json
import hashlib
import time
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
@dataclass
class MemoryItem:
    """单条记忆项"""
    id: str
    content: str
    source: str = "user"               # user / system / extracted
    timestamp: float = field(default_factory=time.time)
    version: int = 1                    # 用于增量更新版本控制
    is_active: bool = True              # 是否有效（冲突时标记过期）
    weight: float = 1.0                 # 权重，用于冲突排序
    embedding: Optional[List[float]] = None  # 实际应存储向量


class MemoryStore:
    """
    记忆存储器，演示三种更新与合并策略：
    1. 增量更新 - 仅更新变化字段，无需全量重写
    2. 冲突解决 - 检测矛盾，按时间/权重标记过期
    3. 记忆合并 - 融合相似记忆，去重冗余
    """

    def __init__(self):
        self.memories: Dict[str, MemoryItem] = {}
        self.id_counter = 0

    # ---------- 基础 CRUD ----------
    def _generate_id(self, content: str) -> str:
        """基于内容生成唯一 ID（实际可用更健壮的方式）"""
        self.id_counter += 1
        return f"mem_{self.id_counter:04d}"

    def add_memory(self, content: str, source: str = "user") -> MemoryItem:
        """新增记忆"""
        mem_id = self._generate_id(content)
        mem = MemoryItem(id=mem_id, content=content, source=source)
        self.memories[mem_id] = mem
        return mem

    # ---------- 1. 增量更新 ----------
    def update_memory(self, mem_id: str, new_content: str):
        """
        增量更新：仅更新内容字符串并递增版本号，不重写其他字段。
        同时重新计算嵌入（此处省略实际计算）。
        """
        if mem_id not in self.memories:
            raise KeyError(f"Memory {mem_id} not found")
        mem = self.memories[mem_id]
        old_content = mem.content
        mem.content = new_content
        mem.version += 1
        mem.timestamp = time.time()  # 更新时间戳
        # 实际场景应重新生成 embedding（增量更新向量库）
        # mem.embedding = generate_embedding(new_content)
        print(f"[增量更新] 记忆 {mem_id} 内容变更: '{old_content}' -> '{new_content}' (版本 {mem.version})")

    # ---------- 2. 冲突解决 ----------
    def resolve_conflict(self, mem1: MemoryItem, mem2: MemoryItem) -> MemoryItem:
        """
        冲突解决逻辑：
        - 比较时间戳：越新的权重越高
        - 比较来源权重：例如 'explicit' > 'extracted' > 'implicit'
        - 返回优胜者，将失败者标记为过期
        """
        source_weights = {"user_explicit": 3, "user": 2, "extracted": 1, "system": 0}
        w1 = source_weights.get(mem1.source, 1)
        w2 = source_weights.get(mem2.source, 1)

        # 综合得分 = 时间因子 * 来源权重（简化）
        score1 = mem1.timestamp * 0.1 + w1 * 10
        score2 = mem2.timestamp * 0.1 + w2 * 10

        if score1 >= score2:
            winner, loser = mem1, mem2
        else:
            winner, loser = mem2, mem1

        loser.is_active = False
        print(f"[冲突解决] 矛盾记忆: '{mem1.content}' vs '{mem2.content}'")
        print(f"          优胜: '{winner.content}' (来源:{winner.source}, 时间:{datetime.fromtimestamp(winner.timestamp).strftime('%H:%M:%S')})")
        print(f"          淘汰: '{loser.content}' 已标记为过期")
        return winner

    def detect_conflicts(self, field: str) -> List[tuple]:
        """
        检测指定字段的矛盾记忆（简单示例：基于关键词匹配）
        实际应使用语义相似度 + 实体关系判断。
        """
        # 这里模拟检测关于 "职业" 的矛盾陈述
        field_keywords = {"职业": ["工程师", "开发者", "程序员", "Java", "Python"],
                         "偏好语言": ["Python", "Java", "Go", "Rust"]}
        groups = {}
        for mem in self.memories.values():
            if not mem.is_active:
                continue
            for kw in field_keywords.get(field, []):
                if kw in mem.content:
                    groups.setdefault(field, []).append(mem)
                    break

        conflicts = []
        if field in groups and len(groups[field]) > 1:
            # 找出内容不同的记忆对
            items = groups[field]
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    if items[i].content != items[j].content:
                        conflicts.append((items[i], items[j]))
        return conflicts

    def auto_resolve_conflicts(self, field: str):
        """自动解决指定字段的所有冲突"""
        conflicts = self.detect_conflicts(field)
        for m1, m2 in conflicts:
            self.resolve_conflict(m1, m2)

    # ---------- 3. 记忆合并 ----------
    def merge_similar_memories(self, mem_ids: List[str]) -> Optional[MemoryItem]:
        """
        将多条相似/相关的记忆合并为一条。
        模拟合并策略：提取共同关键词，生成摘要式记忆。
        实际可用 LLM 进行智能合并。
        """
        if not mem_ids:
            return None

        valid_mems = [self.memories[mid] for mid in mem_ids if mid in self.memories and self.memories[mid].is_active]
        if len(valid_mems) < 2:
            return None

        # 简单合并：拼接内容（实际应调用 LLM 总结）
        contents = [m.content for m in valid_mems]
        merged_content = "；".join(contents)  # 示例：实际可改为 "用户偏好：喜欢简洁代码，需要注释"
        # 更好的合并示例（针对特定模式）
        if all("代码" in c for c in contents):
            merged_content = "用户偏好简洁且带注释的 Python 代码。"

        # 创建新记忆
        new_mem = MemoryItem(
            id=self._generate_id(merged_content),
            content=merged_content,
            source="merged",
            timestamp=max(m.timestamp for m in valid_mems),
            weight=sum(m.weight for m in valid_mems) / len(valid_mems)
        )
        self.memories[new_mem.id] = new_mem

        # 将旧记忆标记为过期
        for m in valid_mems:
            m.is_active = False

        print(f"[记忆合并] 将 {len(valid_mems)} 条相关记忆合并为一条:")
        for m in valid_mems:
            print(f"           - {m.content}")
        print(f"           → 合并结果: {merged_content}")
        return new_mem

    def find_similar_memories(self, threshold: float = 0.7) -> List[List[str]]:
        """
        寻找相似记忆组（模拟，实际用向量相似度聚类）
        这里用简单规则：包含相同关键词超过一定数量。
        """
        groups = []
        active_mems = [m for m in self.memories.values() if m.is_active]
        for i, m1 in enumerate(active_mems):
            for m2 in active_mems[i+1:]:
                # 简单相似判断：共享词数量
                words1 = set(m1.content)
                words2 = set(m2.content)
                if len(words1 & words2) / max(len(words1), len(words2)) > 0.5:
                    groups.append([m1.id, m2.id])
        # 合并有交集的组（简化）
        merged_groups = []
        for pair in groups:
            added = False
            for g in merged_groups:
                if any(pid in g for pid in pair):
                    g.update(pair)
                    added = True
                    break
            if not added:
                merged_groups.append(set(pair))
        return [list(g) for g in merged_groups]

    def auto_merge_all(self):
        """自动寻找并合并所有相似记忆"""
        groups = self.find_similar_memories()
        for g in groups:
            if len(g) >= 2:
                self.merge_similar_memories(g)

    # ---------- 辅助展示 ----------
    def list_active_memories(self) -> List[str]:
        return [m.content for m in self.memories.values() if m.is_active]

    def show_status(self):
        print("\n====== 当前有效记忆 ======")
        for mem in self.memories.values():
            if mem.is_active:
                print(f"- {mem.content} (版本:{mem.version}, 来源:{mem.source})")
        print("==========================\n")


# ========== 演示运行 ==========
if __name__ == "__main__":
    store = MemoryStore()

    print("=== 步骤1：初始化记忆 ===")
    mem1 = store.add_memory("我是 Python 开发者", source="user")
    mem2 = store.add_memory("喜欢简洁的 Python 代码", source="user")
    mem3 = store.add_memory("代码最好带详细注释", source="user")
    mem4 = store.add_memory("我常用 Java 开发后端", source="user")  # 与 mem1 矛盾
    store.show_status()

    print("\n=== 步骤2：增量更新（修改记忆内容）===")
    store.update_memory(mem2.id, "我喜欢非常简洁且符合 PEP8 的 Python 代码")
    store.show_status()

    print("\n=== 步骤3：冲突检测与解决（职业矛盾）===")
    conflicts = store.detect_conflicts("职业")
    if conflicts:
        print(f"检测到 {len(conflicts)} 对冲突记忆")
        store.auto_resolve_conflicts("职业")
    store.show_status()

    print("\n=== 步骤4：记忆合并（相似偏好合并）===")
    # 手动指定要合并的记忆 ID（实际应自动发现）
    similar_ids = [mem2.id, mem3.id]  # 关于代码风格和注释的两条
    store.merge_similar_memories(similar_ids)
    store.show_status()

    print("\n=== 步骤5：再次查看最终有效记忆 ===")
    for content in store.list_active_memories():
        print(f"  • {content}")