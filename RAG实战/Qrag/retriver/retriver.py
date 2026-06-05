import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, models
from typing import List, Tuple


class BM25Retriever:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, float(scores[idx])) for idx in top_indices]


class VectorRetriever:
    def __init__(self, embed_model_path: str = None):
        if embed_model_path:
            try:
                self.model = SentenceTransformer(embed_model_path)
            except Exception:
                word_embedding_model = models.Transformer(embed_model_path)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents: List[str] = []
        self.doc_embeddings = None

    def build_index(self, documents: List[str]):
        self.documents = documents
        self.doc_embeddings = self.model.encode(documents, convert_to_tensor=False).astype('float32')

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_embedding = self.model.encode([query], convert_to_tensor=False).astype('float32')
        similarities = np.linalg.norm(self.doc_embeddings - query_embedding, axis=1)
        top_indices = np.argsort(similarities)[:top_k]
        return [(idx, float(similarities[idx])) for idx in top_indices]


class RRFusion:
    @staticmethod
    def fuse(result_lists: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
        scores = {}
        for results in result_lists:
            for rank, (doc_id, _) in enumerate(results, start=1):
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: VectorRetriever, rrf_k: int = 60):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        bm25_results = self.bm25_retriever.search(query, top_k)
        vector_results = self.vector_retriever.search(query, top_k)
        fused = RRFusion.fuse([bm25_results, vector_results], k=self.rrf_k)
        return fused[:top_k]


class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, float(scores[idx])) for idx in ranked_indices]
