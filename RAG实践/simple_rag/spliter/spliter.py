import abc
import re
from typing import List, Optional
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np


class BaseChunker(abc.ABC):
    """分块器基类：定义统一接口"""

    def __init__(self, chunk_size: int = 512, tokenizer_model: str = "gpt-3.5-turbo"):
        self.chunk_size = chunk_size
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)

    def count_tokens(self, text: str) -> int:
        """计算文本的Token数"""
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
        current_splits = [text]
        for sep in self.separators:
            new_splits = []
            for split in current_splits:
                if self.count_tokens(split) > self.chunk_size:
                    new_splits.extend(self._split_text(split, sep))
                else:
                    new_splits.append(split)
            current_splits = new_splits
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
        if not embedding_model:
            raise ValueError("必须提供本地嵌入模型路径")
        self.embedding_model = SentenceTransformer(embedding_model, local_files_only=True)
        self.similarity_threshold = similarity_threshold

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def split(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[。！？.!?])\s*', text.strip())
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
