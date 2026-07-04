import numpy as np
import re
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Any

# 模拟 embedding 函数，实际应使用 OpenAIEmbeddings / 本地模型
def mock_embed(text: str) -> np.ndarray:
    """生成模拟向量（仅用于演示）"""
    seed = sum(ord(c) for c in text) % 100
    np.random.seed(seed)
    return np.random.rand(128)

# 模拟余弦相似度
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

class MemoryItem:
    """单条记忆项"""
    def __init__(self, id: str, content: str, metadata: Dict[str, Any] = None):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.embedding = mock_embed(content)  # 实际应调用 embed_model.embed_query(content)


class MemoryRetriever:
    """
    记忆检索器，演示：
    1. 混合检索（向量语义 + BM25 关键词）
    2. 二次重排（Cross-Encoder Rerank）
    3. 分层召回（热缓存 → 向量库 → 冷存储）
    4. 动态注入（根据任务复杂度调整数量）
    """

    def __init__(self):
        # 模拟向量库（长期语义记忆）
        self.vector_store: List[MemoryItem] = []
        # 模拟 BM25 所需倒排索引（极简版）
        self.inverted_index: Dict[str, set] = {}
        # 模拟热缓存（当前会话高频记忆）
        self.hot_cache: OrderedDict[str, MemoryItem] = OrderedDict()
        # 模拟冷存储（历史归档，仅通过 ID 获取）
        self.cold_storage: Dict[str, MemoryItem] = {}

        # 初始化一些模拟记忆数据
        self._init_demo_memories()

    def _init_demo_memories(self):
        """添加演示记忆数据"""
        memories = [
            ("mem1", "用户姓名王小明，职业 Python 工程师，喜欢咖啡。"),
            ("mem2", "用户需要爬取豆瓣电影 Top250，使用 requests 和 BeautifulSoup。"),
            ("mem3", "用户已解决豆瓣反爬问题，设置 User-Agent 和 Referer 头。"),
            ("mem4", "用户偏好 IDE 是 PyCharm，喜欢深色主题。"),
            ("mem5", "上次会话讨论过 Scrapy 框架与异步爬虫的区别。"),
            ("mem6", "用户所在城市是北京，通勤喜欢听播客。"),
        ]
        for mid, content in memories:
            item = MemoryItem(mid, content)
            self.vector_store.append(item)
            self._add_to_inverted_index(mid, content)

    def _add_to_inverted_index(self, doc_id: str, text: str):
        """构建简易 BM25 倒排索引（仅分词）"""
        words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", text.lower()))
        for w in words:
            self.inverted_index.setdefault(w, set()).add(doc_id)

    # ---------- 1. 混合检索：向量 + BM25 ----------
    def _vector_search(self, query: str, top_k: int = 20) -> List[Tuple[MemoryItem, float]]:
        """纯向量检索"""
        q_emb = mock_embed(query)
        scores = []
        for item in self.vector_store:
            sim = cosine_sim(q_emb, item.embedding)
            scores.append((item, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[MemoryItem, float]]:
        """简易 BM25 关键词检索（模拟）"""
        query_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", query.lower()))
        doc_scores = {}
        for w in query_words:
            if w in self.inverted_index:
                for doc_id in self.inverted_index[w]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1
        # 找到对应的 MemoryItem
        id_to_item = {item.id: item for item in self.vector_store}
        results = []
        for doc_id, score in doc_scores.items():
            if doc_id in id_to_item:
                results.append((id_to_item[doc_id], float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.7) -> List[MemoryItem]:
        """
        混合检索：向量权重 α，BM25 权重 1-α。
        实际可使用 Weaviate / Elasticsearch 的混合检索功能。
        """
        vec_results = self._vector_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)

        # 归一化分数并融合
        vec_dict = {item.id: (item, score) for item, score in vec_results}
        bm25_dict = {item.id: (item, score) for item, score in bm25_results}

        # 归一化（简化：各自除以最大分）
        max_vec = max((s for _, s in vec_results), default=1)
        max_bm25 = max((s for _, s in bm25_results), default=1)

        fused = {}
        all_ids = set(vec_dict.keys()) | set(bm25_dict.keys())
        for mid in all_ids:
            vec_score = vec_dict[mid][1] / max_vec if mid in vec_dict else 0
            bm25_score = bm25_dict[mid][1] / max_bm25 if mid in bm25_dict else 0
            fused[mid] = alpha * vec_score + (1 - alpha) * bm25_score

        sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:top_k]
        return [vec_dict.get(mid, bm25_dict.get(mid))[0] for mid in sorted_ids]

    # ---------- 2. 二次重排 (Rerank) ----------
    def rerank(self, query: str, candidates: List[MemoryItem], top_k: int = 5) -> List[MemoryItem]:
        """
        使用 Cross-Encoder 进行精细重排。
        这里模拟一个简单的关键词重叠度作为 rerank 分数。
        真实场景应使用 Cohere Rerank / HuggingFace CrossEncoder。
        """
        query_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", query.lower()))
        scores = []
        for item in candidates:
            content_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z]+", item.content.lower()))
            overlap = len(query_words & content_words)
            # 模拟交叉编码器会考虑语义顺序等，这里简单加分
            score = overlap + (0.1 * len(content_words))  # 防止零分
            scores.append((item, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scores[:top_k]]

    # ---------- 3. 分层召回 ----------
    def layered_recall(self, query: str, session_id: str = "current") -> List[MemoryItem]:
        """
        分层召回：先查热缓存，再查向量库混合检索，最后考虑冷存储。
        """
        recalled = []

        # 第一层：热缓存（如 Redis 存储的当前会话记忆）
        # 模拟：缓存中只保留最近 3 条交互
        cache_items = list(self.hot_cache.values())
        recalled.extend(cache_items)
        print(f"[分层召回] 热缓存命中 {len(cache_items)} 条记忆")

        # 第二层：向量库混合检索（补齐到期望数量）
        needed = 10 - len(recalled)
        if needed > 0:
            hybrid_results = self.hybrid_search(query, top_k=needed)
            # 去重（避免热缓存中已有的）
            existing_ids = {item.id for item in recalled}
            for item in hybrid_results:
                if item.id not in existing_ids:
                    recalled.append(item)
                    existing_ids.add(item.id)
            print(f"[分层召回] 向量库补充 {len(hybrid_results)} 条记忆")

        # 第三层：冷存储（仅当向量库结果不足时，实际可异步获取）
        if len(recalled) < 5:
            # 模拟从冷存储捞取最旧的一条
            if self.cold_storage:
                oldest = list(self.cold_storage.values())[0]
                recalled.append(oldest)
                print(f"[分层召回] 冷存储补充 1 条记忆")
        return recalled

    # ---------- 4. 动态注入 ----------
    def retrieve_for_task(self, query: str, task_complexity: str = "simple") -> str:
        """
        根据任务复杂度动态决定召回数量并生成注入上下文的文本。
        """
        # 先分层召回粗排候选（最多 10 条）
        candidates = self.layered_recall(query)
        print(f"[粗召回] 获得 {len(candidates)} 条候选记忆")

        # 二次重排
        reranked = self.rerank(query, candidates, top_k=10)
        print(f"[重排后] 排序完成，Top5 分数最高记忆如下：")
        for i, item in enumerate(reranked[:5]):
            print(f"  {i+1}. {item.content}")

        # 动态注入数量决策
        inject_map = {
            "simple": 2,      # 简单问答只给 Top2
            "moderate": 5,    # 中等复杂给 Top5
            "complex": 8,     # 复杂任务给 Top8
        }
        top_n = inject_map.get(task_complexity, 3)
        final_memories = reranked[:top_n]

        # 构造注入上下文的字符串
        context_lines = ["[相关记忆]"]
        for i, mem in enumerate(final_memories, 1):
            context_lines.append(f"{i}. {mem.content}")
        context = "\n".join(context_lines)
        print(f"[动态注入] 任务复杂度 '{task_complexity}'，注入 {top_n} 条记忆，Token 预估：{len(context)} 字符")
        return context

    # ---------- 辅助：更新热缓存 ----------
    def add_to_hot_cache(self, item: MemoryItem):
        """模拟将最新记忆放入 Redis 热缓存"""
        self.hot_cache[item.id] = item
        # 保持缓存大小
        if len(self.hot_cache) > 3:
            self.hot_cache.popitem(last=False)

    # ---------- 辅助：归档到冷存储 ----------
    def archive_to_cold(self, item_id: str):
        if item_id in {m.id for m in self.vector_store}:
            for m in self.vector_store:
                if m.id == item_id:
                    self.cold_storage[item_id] = m
                    break


# ========== 演示运行 ==========
if __name__ == "__main__":
    retriever = MemoryRetriever()

    # 模拟当前会话热缓存（用户刚提到的内容）
    session_mem = MemoryItem("s1", "用户刚刚说想用 Python 爬取豆瓣电影评论，已经安装了 requests。")
    retriever.add_to_hot_cache(session_mem)

    # 模拟查询
    user_query = "豆瓣爬虫反爬怎么处理？"

    print("=" * 50)
    print(f"用户查询: {user_query}\n")

    # 场景1：简单问答，注入较少记忆
    print("--- 场景：简单问答 ---")
    context_simple = retriever.retrieve_for_task(user_query, task_complexity="simple")
    print("\n注入上下文：")
    print(context_simple)

    print("\n" + "=" * 50 + "\n")

    # 场景2：复杂任务，需要更多上下文
    print("--- 场景：复杂任务（如代码生成）---")
    context_complex = retriever.retrieve_for_task(user_query, task_complexity="complex")
    print("\n注入上下文：")
    print(context_complex)