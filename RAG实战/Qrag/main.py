from typing import List, Dict, Optional
from knowledge_base.kb_faiss import EmbeddingModel, VectorStore
from spliter.spliter import TextSplitter
from retriver.retriver import BM25Retriever, VectorRetriever, HybridRetriever, Reranker


class RAGPipeline:
    def __init__(
        self,
        embed_model_path: str = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        chunk_strategy: str = "recursive",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.embed_model = EmbeddingModel(model_path=embed_model_path)
        self.splitter = TextSplitter(
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        self.reranker = Reranker(model_name=reranker_model)
        self.vector_store: Optional[VectorStore] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.chunks: List[str] = []

    def build_index(self, texts: List[str]):
        self.chunks = self.splitter.split_texts(texts)
        embeddings = self.embed_model.encode(self.chunks, show_progress=True)
        self.vector_store = VectorStore(dimension=embeddings.shape[1])
        self.vector_store.build_index(self.chunks, embeddings)
        self.bm25_retriever = BM25Retriever(self.chunks)

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
        results = hybrid_retriever.search(query, top_k=top_k * 2)

        retrieved_texts = [self.chunks[idx] for idx, _ in results]
        reranked = self.reranker.rerank(query, retrieved_texts, top_k=top_k)

        return [
            {"text": self.chunks[results[idx][0]], "score": float(score)}
            for idx, (_, score) in enumerate(reranked)
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
        embed_model_path: str = None,
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

    def query(self, question: str, top_k: int = 3):
        return self.pipeline.query(question, top_k=top_k)


if __name__ == "__main__":
    rag = SimpleRAG(
        embed_model_path=r"E:\Qagent\models\allMiniLML6v2",
        chunk_size=200,
        overlap=30
    )

    docs = [
        "RAG（检索增强生成）是一种结合检索系统和生成模型的技术。",
        "它通过从外部知识库检索相关信息来增强大语言模型的能力。",
        "向量数据库如FAISS、Milvus等用于高效存储和检索向量嵌入。",
        "文本分块是将长文档切分成小片段的重要预处理步骤。"
    ]

    rag.build_index(docs)

    result = rag.query("什么是RAG技术？")
    print("=" * 50)
    print(f"问题: {result['question']}")
    print(f"\n回答:\n{result['answer']}")
    print(f"\n参考来源: {len(result['references'])}条")
