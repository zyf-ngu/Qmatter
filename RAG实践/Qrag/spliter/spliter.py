import abc
import re
from typing import List, Optional
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from mistune import markdown as parse_markdown


class BaseChunker(abc.ABC):
    """分块器基类：定义统一接口"""

    def __init__(self, chunk_size: int = 512, tokenizer_model: str = "gpt-3.5-turbo"):
        self.chunk_size = chunk_size
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)

    def count_tokens(self, text: str) -> int:
        """计算文本的Token数"""
        # 分词（Tokenization）：根据分词器的词汇表规则，将输入文本切分成最小的语义单元（如单词、子词或字符）。
        # 编码（Encoding）：将每个token 映射到词汇表中对应的唯一整数ID。
        return len(self.tokenizer.encode(text))

    @abc.abstractmethod
    def split(self, text: str) -> List[str]:
        """核心分块方法：输入原始文本，输出文本块列表"""
        pass


class FixedSizeChunker(BaseChunker):
    """固定长度分块器（支持重叠窗口）"""

    def __init__(self, chunk_size: int = 512, overlap: int = 100, **kwargs):
        super().__init__(chunk_size, **kwargs)
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            start += self.chunk_size - self.overlap
        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """递归字符分块器（按语义分隔符优先级切分）"""

    def __init__(
        self,
        chunk_size: int = 512,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(chunk_size, **kwargs)
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", " ", ""]

    def _split_text(self, text: str, separator: str) -> List[str]:
        if not separator:
            return list(text)
        splits = text.split(separator)
        return [s + separator for s in splits[:-1]] + ([splits[-1]] if splits[-1] else [])

    def _merge_splits(self, splits: List[str]) -> List[str]:
        chunks = []
        current_chunk = ""
        for split in splits:
            if self.count_tokens(current_chunk + split) <= self.chunk_size:
                current_chunk += split
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if self.count_tokens(split) > self.chunk_size:
                    chunks.extend(self._merge_splits(self._split_text(split, self.separators[-1])))
                else:
                    current_chunk = split
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def split(self, text: str) -> List[str]:
        final_chunks = []
        # 最初是完整的文本
        current_splits = [text]
        for sep in self.separators:
            new_splits = []
            for split in current_splits:
                if self.count_tokens(split) > self.chunk_size:
                    new_splits.extend(self._split_text(split, sep))
                else:
                    new_splits.append(split)
            # 每一次遍历分隔符都会更新分割后的文本
            current_splits = new_splits
        # 把所有分割符遍历完后 长度过小的的文本融合在一起
        return self._merge_splits(current_splits)


class SemanticChunker(BaseChunker):
    """语义分块器（基于嵌入向量相似度）"""

    def __init__(
        self,
        chunk_size: int = 512,
        embedding_model: str = r"E:\Qagent\models\allMiniLML6v2",
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(chunk_size, **kwargs)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def split(self, text: str) -> List[str]:
        # Simple sentence splitter for Chinese/English text
        import re
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[。！？.!?])\s*', text.strip())
        # Remove empty strings
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return []
        embeddings = self.embedding_model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(current_embedding, embeddings[i])
            if sim >= self.similarity_threshold and self.count_tokens("".join(current_chunk + [sentences[i]])) <= self.chunk_size:
                current_chunk.append(sentences[i])
                current_embedding = np.mean(embeddings[:i+1], axis=0)
            else:
                chunks.append("".join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
        if current_chunk:
            chunks.append("".join(current_chunk))
        return chunks


class MarkdownStructureChunker(BaseChunker):
    """Markdown结构感知分块器（按标题层级切分）"""

    def __init__(self, chunk_size: int = 512, **kwargs):
        super().__init__(chunk_size, **kwargs)
        self._level_pattern = re.compile(r"^(#{1,6})\s")

    def _parse_markdown(self, text: str) -> List[dict]:
        lines = text.split("\n")
        elements = []
        current_heading = ""
        current_content = []
        for line in lines:
            match = self._level_pattern.match(line)
            if match:
                if current_heading or current_content:
                    elements.append({"heading": current_heading, "content": "\n".join(current_content).strip()})
                current_heading = line.strip()
                current_content = []
            else:
                current_content.append(line)
        if current_heading or current_content:
            elements.append({"heading": current_heading, "content": "\n".join(current_content).strip()})
        return elements

    def split(self, text: str) -> List[str]:
        elements = self._parse_markdown(text)
        chunks = []
        current_chunk = ""
        for elem in elements:
            block = f"{elem['heading']}\n{elem['content']}".strip()
            if self.count_tokens(current_chunk + "\n" + block) <= self.chunk_size:
                current_chunk = f"{current_chunk}\n{block}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if self.count_tokens(block) > self.chunk_size:
                    sub_chunker = RecursiveCharacterChunker(chunk_size=self.chunk_size)
                    chunks.extend(sub_chunker.split(block))
                else:
                    current_chunk = block
        if current_chunk:
            chunks.append(current_chunk)
        return chunks


class RuleBasedChunker(BaseChunker):
    """规则分块器（基于自定义正则表达式）"""

    def __init__(self, chunk_size: int = 512, pattern: str = r"第\d+条", **kwargs):
        super().__init__(chunk_size, **kwargs)
        self.pattern = re.compile(pattern)

    def split(self, text: str) -> List[str]:
        splits = self.pattern.split(text)
        matches = self.pattern.findall(text)
        chunks = []
        current_chunk = ""
        for i in range(len(splits)):
            block = (matches[i-1] if i > 0 else "") + splits[i]
            if self.count_tokens(current_chunk + block) <= self.chunk_size:
                current_chunk += block
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if self.count_tokens(block) > self.chunk_size:
                    sub_chunker = RecursiveCharacterChunker(chunk_size=self.chunk_size)
                    chunks.extend(sub_chunker.split(block))
                else:
                    current_chunk = block
        if current_chunk:
            chunks.append(current_chunk)
        return chunks


class TextSplitter:
    def __init__(self, strategy: str = "recursive", chunk_size: int = 512, overlap: int = 50, **kwargs):
        self.strategy = strategy
        if strategy == "fixed":
            self.splitter = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap, **kwargs)
        elif strategy == "semantic":
            self.splitter = SemanticChunker(chunk_size=chunk_size, **kwargs)
        elif strategy == "markdown":
            self.splitter = MarkdownStructureChunker(chunk_size=chunk_size, **kwargs)
        elif strategy == "rule":
            pattern = kwargs.get("pattern", r"第\d+条")
            self.splitter = RuleBasedChunker(chunk_size=chunk_size, pattern=pattern, **kwargs)
        else:
            self.splitter = RecursiveCharacterChunker(chunk_size=chunk_size, **kwargs)

    def split_texts(self, texts: List[str]) -> List[str]:
        all_chunks = []
        for text in texts:
            if text.strip():
                chunks = self.splitter.split(text)
                all_chunks.extend(chunks)
        return all_chunks

    def split(self, text: str) -> List[str]:
        return self.splitter.split(text)


if __name__=='__main__':
    sample_text = """
    # XX智能扫地机器人手册
    ## 一、核心功能
    XX智能扫地机器人具备LDS激光导航功能，可实现10米范围内的精准建图，支持全屋分区清扫、禁区设置，单次续航可达180分钟，适配150㎡以内的户型。同时搭载2700Pa大吸力，可清理浮尘、毛发、颗粒物，支持自动集尘，30天无需手动倒垃圾。

    ## 二、故障排查
    ### 2.1 无法建图
    请检查激光雷达是否有遮挡，重启设备后重试。
    ### 2.2 吸力不足
    请清理滚刷与尘盒，检查吸力档位是否设置正确。
    """
    # 1. 固定长度分块print("=== 固定长度分块 ===")
    fixed_chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    for i, chunk in enumerate(fixed_chunker.split(sample_text)):
        print(f"块{i + 1}:\n{chunk}\n")
    # 2. 递归字符分块print("=== 递归字符分块 ===")
    recursive_chunker = RecursiveCharacterChunker(chunk_size=200)
    for i, chunk in enumerate(recursive_chunker.split(sample_text)):
        print(f"块{i + 1}:\n{chunk}\n")
    # 3. 语义分块print("=== 语义分块 ===")
    semantic_chunker = SemanticChunker(chunk_size=200, similarity_threshold=0.7)
    for i, chunk in enumerate(semantic_chunker.split(sample_text)):
        print(f"块{i + 1}:\n{chunk}\n")
    # 4. Markdown结构分块print("=== Markdown结构分块 ===")
    md_chunker = MarkdownStructureChunker(chunk_size=300)
    for i, chunk in enumerate(md_chunker.split(sample_text)):
        print(f"块{i + 1}:\n{chunk}\n")
    # 5. 规则分块（以法律条文为例）
    legal_text = """
    第五百零九条 当事人应当按照约定全面履行自己的义务。当事人应当遵循诚信原则，根据合同的性质、目的和交易习惯履行通知、协助、保密等义务。
    第五百一十条 合同生效后，当事人就质量、价款或者报酬、履行地点等内容没有约定或者约定不明确的，可以协议补充；不能达成补充协议的，按照合同相关条款或者交易习惯确定。
    """
    print("=== 规则分块（法律条文） ===")
    rule_chunker = RuleBasedChunker(chunk_size=200, pattern=r"第\d+条")
    for i, chunk in enumerate(rule_chunker.split(legal_text)):
        print(f"块{i + 1}:\n{chunk}\n")