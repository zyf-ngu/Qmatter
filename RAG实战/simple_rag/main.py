import os
from typing import List, Dict, Optional
from knowledge_base.kb_faiss import EmbeddingModel, VectorStore
from spliter.spliter import TextSplitter
from retriver.retriver import BM25Retriever, VectorRetriever, HybridRetriever, Reranker
from parser import DocumentLoader


class RAGPipeline:
    def __init__(
        self,
        embed_model_path: str = r"E:\Qagent\models\allMiniLML6v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        chunk_strategy: str = "recursive",
        reranker_model_path: str = None
    ):
        self.embed_model = EmbeddingModel(model_path=embed_model_path)
        self.splitter = TextSplitter(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        self.doc_loader = DocumentLoader()
        self.reranker = None
        if reranker_model_path:
            self.reranker = Reranker(model_path=reranker_model_path)
        self.vector_store: Optional[VectorStore] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.chunks: List[str] = []

    def build_index(self, texts: List[str]):
        self.chunks = self.splitter.split_texts(texts)
        embeddings = self.embed_model.encode(self.chunks, show_progress=True)
        self.vector_store = VectorStore(dimension=embeddings.shape[1])
        self.vector_store.build_index(self.chunks, embeddings)
        self.bm25_retriever = BM25Retriever(self.chunks)

    def build_index_from_files(self, file_paths: List[str]):
        all_texts = []
        doc_results = self.doc_loader.load_files(file_paths)
        for result in doc_results:
            content = result["content"]
            if content:
                combined_text = f"【{os.path.basename(result['file_path'])}】\n" + "\n".join(content)
                all_texts.append(combined_text)
        self.build_index(all_texts)

    def build_index_from_directory(self, dir_path: str, recursive: bool = False):
        all_texts = []
        doc_results = self.doc_loader.load_directory(dir_path, recursive)
        for result in doc_results:
            content = result["content"]
            if content:
                combined_text = f"【{os.path.basename(result['file_path'])}】\n" + "\n".join(content)
                all_texts.append(combined_text)
        self.build_index(all_texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.vector_store or not self.bm25_retriever:
            raise RuntimeError("请先调用 build_index() 构建索引")

        vector_retriever = VectorRetriever()
        vector_retriever.build_index(self.chunks)

        hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            vector_retriever=vector_retriever,
            rrf_k=60
        )
        results = hybrid_retriever.search(query, top_k=top_k * 2 if self.reranker else top_k)

        if self.reranker:
            retrieved_texts = [self.chunks[idx] for idx, _ in results]
            reranked = self.reranker.rerank(query, retrieved_texts, top_k=top_k)
            return [
                {"text": self.chunks[results[idx][0]], "score": float(score)}
                for idx, (_, score) in enumerate(reranked)
            ]
        else:
            return [
                {"text": self.chunks[idx], "score": float(score)}
                for idx, score in results
            ]

    def query(self, question: str, top_k: int = 3) -> Dict:
        references = self.retrieve(question, top_k=top_k)
        context = "\n".join([f"[{i+1}] {ref['text']}" for i, ref in enumerate(references)])
        answer = f"基于{len(references)}条参考内容回答：\n\n{context}"

        return {
            "question": question,
            "answer": answer,
            "references": references
        }


class SimpleRAG:
    def __init__(
        self,
        embed_model_path: str = r"E:\Qagent\models\allMiniLML6v2",
        chunk_size: int = 512,
        overlap: int = 50
    ):
        self.pipeline = RAGPipeline(
            embed_model_path=embed_model_path,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def build_index(self, texts: List[str]):
        self.pipeline.build_index(texts)

    def build_index_from_files(self, file_paths: List[str]):
        self.pipeline.build_index_from_files(file_paths)

    def build_index_from_directory(self, dir_path: str, recursive: bool = False):
        self.pipeline.build_index_from_directory(dir_path, recursive)

    def query(self, question: str, top_k: int = 3):
        return self.pipeline.query(question, top_k=top_k)


if __name__ == "__main__":
    print("=" * 60)
    print("  简单 RAG 系统演示")
    print("=" * 60)

    print("\n1. 初始化 RAG 系统...")
    rag = SimpleRAG(
        embed_model_path=r"E:\Qagent\models\allMiniLML6v2",
        chunk_size=400,
        overlap=50
    )
    print("   ✓ 初始化完成")

    sample_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    print(f"\n2. 从目录加载文档: {sample_dir}")

    if os.path.exists(sample_dir):
        files = [f for f in os.listdir(sample_dir) if not f.endswith('.py')]
        for f in files:
            print(f"   - {f}")

        print("\n3. 构建向量索引...")
        rag.build_index_from_directory(sample_dir)
        print(f"   ✓ 索引构建完成，共 {len(rag.pipeline.chunks)} 个块")

        print("\n4. 开始查询问答...")
        queries = [
            "文档处理平台的版本是多少？",
            "文档解析器支持哪些文档格式？",
            "公司成立时间？"
        ]

        for i, q in enumerate(queries, 1):
            print(f"\n--- 查询 {i}: {q}")
            result = rag.query(q, top_k=3)
            print(f"\n回答:\n{result['answer']}")
    else:
        print(f"   ⚠ 目录不存在: {sample_dir}")

    print("\n" + "=" * 60)
    print("  演示结束")
    print("=" * 60)
