import time
import math
import heapq
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class MemoryItem:
    """单条记忆项，包含评分所需字段"""
    id: str
    content: str
    importance: float = 1.0          # 重要度 (0~1)
    access_count: int = 0            # 访问频次
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    redundancy: float = 0.0          # 冗余度 (0~1，越低越好)
    is_archived: bool = False        # 是否已归档
    is_deleted: bool = False         # 是否已删除

    def access(self):
        """模拟一次访问"""
        self.access_count += 1
        self.last_access = time.time()


class MemoryManagerWithForgetting:
    """
    记忆管理器，演示遗忘与压缩策略：
    1. 时间衰减遗忘（艾宾浩斯半衰期）
    2. 多因子评分遗忘（行业主流）
    3. 分层淘汰机制（热缓存 LRU + 冷存储阈值淘汰）
    """

    def __init__(self):
        # 主存储（向量库 / 长期记忆）
        self.main_storage: Dict[str, MemoryItem] = {}
        # 热缓存（模拟 Redis LRU，容量有限）
        self.hot_cache: OrderedDict[str, MemoryItem] = OrderedDict()
        self.hot_cache_capacity = 3
        # 归档存储（冷数据）
        self.cold_storage: Dict[str, MemoryItem] = {}
        # 删除队列（软删除记录）
        self.deleted_ids: set = set()

        self.id_counter = 0

    def _generate_id(self) -> str:
        self.id_counter += 1
        return f"mem_{self.id_counter:04d}"

    def add_memory(self, content: str, importance: float = 0.5) -> MemoryItem:
        """新增记忆"""
        mem = MemoryItem(
            id=self._generate_id(),
            content=content,
            importance=max(0.0, min(1.0, importance))
        )
        self.main_storage[mem.id] = mem
        self._update_hot_cache(mem)  # 新记忆加入热缓存
        return mem

    def _update_hot_cache(self, mem: MemoryItem):
        """更新热缓存（模拟 LRU 插入）"""
        # 移到最后表示最新访问
        self.hot_cache.pop(mem.id, None)
        self.hot_cache[mem.id] = mem
        # LRU 淘汰：如果超出容量，移除最久未访问的
        while len(self.hot_cache) > self.hot_cache_capacity:
            oldest_id, oldest_mem = self.hot_cache.popitem(last=False)
            print(f"[热缓存 LRU 淘汰] 记忆 '{oldest_mem.content[:20]}...' 移出热缓存")

    # ---------- 1. 时间衰减遗忘（艾宾浩斯半衰期）----------
    def time_decay_score(self, mem: MemoryItem, half_life_days: float = 7.0) -> float:
        """
        基于半衰期计算时间衰减因子 (0~1)。
        公式：score = 0.5 ^ (days_since_access / half_life_days)
        """
        now = time.time()
        days_since_access = (now - mem.last_access) / (24 * 3600)
        decay = 0.5 ** (days_since_access / half_life_days)
        return decay

    # ---------- 2. 多因子评分遗忘（行业主流）----------
    def compute_memory_score(self, mem: MemoryItem) -> float:
        """
        记忆最终评分 = 0.4×重要度 + 0.3×归一化访问频次 - 0.2×时间衰减损失 - 0.1×冗余度
        注意：时间衰减损失 = 1 - 时间衰减因子（即遗忘程度）
        """
        # 访问频次归一化（假设最高访问次数为10，实际可动态计算）
        max_access = max(1, max((m.access_count for m in self.main_storage.values()), default=1))
        norm_access = mem.access_count / max_access

        # 时间衰减因子
        decay_factor = self.time_decay_score(mem)
        # 遗忘损失：1 - 衰减因子
        decay_loss = 1.0 - decay_factor

        score = (0.4 * mem.importance +
                 0.3 * norm_access -
                 0.2 * decay_loss -
                 0.1 * mem.redundancy)
        return score

    def forget_by_score_threshold(self, threshold: float = 0.3):
        """
        扫描主存储，评分低于阈值的记忆执行遗忘操作：
        - 归档：评分低于 threshold 但高于 threshold/2
        - 删除：评分低于 threshold/2
        """
        archived = []
        deleted = []
        for mem in list(self.main_storage.values()):
            score = self.compute_memory_score(mem)
            if score < threshold:
                if score < threshold / 2:
                    # 硬删除
                    self.main_storage.pop(mem.id, None)
                    self.hot_cache.pop(mem.id, None)
                    self.deleted_ids.add(mem.id)
                    mem.is_deleted = True
                    deleted.append(mem)
                else:
                    # 归档到冷存储
                    self.main_storage.pop(mem.id, None)
                    self.hot_cache.pop(mem.id, None)
                    self.cold_storage[mem.id] = mem
                    mem.is_archived = True
                    archived.append(mem)

        if archived:
            print(f"[多因子遗忘] 归档 {len(archived)} 条低价值记忆到冷存储")
        if deleted:
            print(f"[多因子遗忘] 删除 {len(deleted)} 条极低价值记忆")

    # ---------- 3. 分层淘汰机制 ----------
    def hot_cache_lru_eviction(self):
        """热缓存已在上面的 _update_hot_cache 中自动 LRU 淘汰"""
        pass

    def cold_storage_cleanup(self, score_threshold: float = 0.1):
        """清理冷存储中评分过低的记忆（彻底删除）"""
        to_delete = []
        for mem in self.cold_storage.values():
            score = self.compute_memory_score(mem)
            if score < score_threshold:
                to_delete.append(mem.id)
        for mid in to_delete:
            mem = self.cold_storage.pop(mid)
            mem.is_deleted = True
            self.deleted_ids.add(mid)
        if to_delete:
            print(f"[冷存储清理] 删除 {len(to_delete)} 条超低分归档记忆")

    def time_based_auto_archive(self, archive_days: int = 30, delete_days: int = 90):
        """
        基于时间的自动归档/删除（不依赖评分，纯时间策略）
        archive_days: 超过多少天未访问则归档
        delete_days: 超过多少天未访问则删除
        """
        now = time.time()
        archive_threshold = now - archive_days * 24 * 3600
        delete_threshold = now - delete_days * 24 * 3600

        archived = []
        deleted = []
        for mem in list(self.main_storage.values()):
            if mem.last_access < delete_threshold:
                self.main_storage.pop(mem.id, None)
                self.hot_cache.pop(mem.id, None)
                self.deleted_ids.add(mem.id)
                mem.is_deleted = True
                deleted.append(mem)
            elif mem.last_access < archive_threshold:
                self.main_storage.pop(mem.id, None)
                self.hot_cache.pop(mem.id, None)
                self.cold_storage[mem.id] = mem
                mem.is_archived = True
                archived.append(mem)

        if archived:
            print(f"[时间衰减归档] {len(archived)} 条记忆超过 {archive_days} 天未访问，已归档")
        if deleted:
            print(f"[时间衰减删除] {len(deleted)} 条记忆超过 {delete_days} 天未访问，已删除")

    # ---------- 辅助：访问记忆（模拟检索触发）----------
    def access_memory(self, mem_id: str):
        """访问某条记忆，更新频次和热缓存"""
        mem = self.main_storage.get(mem_id) or self.cold_storage.get(mem_id)
        if not mem:
            print(f"记忆 {mem_id} 不存在或已删除")
            return
        mem.access()
        # 如果记忆在冷存储中，且被频繁访问，可以重新提升到主存储（演示不实现）
        if mem_id in self.main_storage:
            self._update_hot_cache(mem)
        print(f"[访问] 记忆 '{mem.content[:30]}...' 访问次数+1")

    # ---------- 状态展示 ----------
    def show_storage_status(self):
        print("\n====== 存储状态 ======")
        print(f"主存储 (活跃记忆): {len(self.main_storage)} 条")
        for mem in self.main_storage.values():
            score = self.compute_memory_score(mem)
            print(f"  - [{mem.id}] {mem.content[:40]}... | 评分:{score:.3f} | 访问:{mem.access_count} | 重要度:{mem.importance:.2f}")
        print(f"热缓存 (容量{self.hot_cache_capacity}): {list(self.hot_cache.keys())}")
        print(f"冷存储 (归档记忆): {len(self.cold_storage)} 条")
        print(f"已删除 ID: {list(self.deleted_ids)[:5]}")
        print("======================\n")


# ========== 演示运行 ==========
if __name__ == "__main__":
    mgr = MemoryManagerWithForgetting()

    print("=== 步骤1：添加初始记忆 ===")
    mgr.add_memory("用户姓名王小明，职业 Python 工程师", importance=0.8)
    mgr.add_memory("喜欢喝咖啡，每天两杯", importance=0.4)
    mgr.add_memory("上次会议讨论爬虫项目进展", importance=0.6)
    mgr.add_memory("昨天天气不错，去公园散步", importance=0.2)
    mgr.add_memory("需要购买 Python 相关书籍", importance=0.7)
    mgr.show_storage_status()

    print("=== 步骤2：模拟访问部分记忆（增加访问频次）===")
    # 模拟频繁访问重要记忆
    for _ in range(3):
        mgr.access_memory("mem_0001")  # 职业信息
    for _ in range(2):
        mgr.access_memory("mem_0003")  # 爬虫项目
    mgr.access_memory("mem_0005")      # 购书
    mgr.show_storage_status()

    print("=== 步骤3：手动调整时间戳模拟长时间未访问（用于测试时间衰减）===")
    # 将一些记忆的最后访问时间设为 30 天前
    now = time.time()
    mgr.main_storage["mem_0002"].last_access = now - 30 * 24 * 3600  # 喝咖啡
    mgr.main_storage["mem_0004"].last_access = now - 60 * 24 * 3600  # 散步
    print("已将 mem_0002 和 mem_0004 的 last_access 设置为很久以前")

    print("\n=== 步骤4：执行基于时间的自动归档/删除 (30天归档, 90天删除) ===")
    mgr.time_based_auto_archive(archive_days=30, delete_days=90)
    mgr.show_storage_status()

    print("=== 步骤5：调整冗余度并执行多因子评分遗忘 (阈值0.3) ===")
    # 设置冗余度：比如 "喜欢喝咖啡" 被认为冗余度较高
    if "mem_0002" in mgr.cold_storage:
        mgr.cold_storage["mem_0002"].redundancy = 0.9
    # 新增一条低价值记忆用于测试删除
    low_mem = mgr.add_memory("今天中午吃了面条", importance=0.1)
    low_mem.access_count = 0
    low_mem.last_access = now - 10 * 24 * 3600
    low_mem.redundancy = 0.8
    mgr.main_storage[low_mem.id] = low_mem  # 放回主存储（因为 add 已在主存储）
    mgr.forget_by_score_threshold(threshold=0.3)
    mgr.show_storage_status()

    print("=== 步骤6：清理冷存储中极低评分记忆 ===")
    mgr.cold_storage_cleanup(score_threshold=0.1)
mgr.show_storage_status()