import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, models
from typing import List, Dict, Optional


class EmbeddingModel:
    def __init__(self, model_path: str = None):
        if model_path:
            try:
                self.model = SentenceTransformer(model_path)
            except Exception as e:
                word_embedding_model = models.Transformer(model_path)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)


class VectorStore:
    def __init__(self, dimension: int = None, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path
        self.index: Optional[faiss.IndexIDMap] = None
        self.custom_id_to_text: Dict[int, str] = {}
        self.custom_id_to_embedding: Dict[int, np.ndarray] = {}
        self.embeddings: Optional[np.ndarray] = None
        self._initialized = False

    def build_index(self, texts: List[str], embeddings: np.ndarray, start_id: int = 0):
        self.embeddings = embeddings.astype('float32')
        self.dimension = embeddings.shape[1]
        custom_ids = np.array([start_id + i for i in range(len(texts))]).astype('int64')
        for i, text in enumerate(texts):
            self.custom_id_to_text[start_id + i] = text
            self.custom_id_to_embedding[start_id + i] = embeddings[i]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(self.embeddings, custom_ids)
        self._initialized = True

    def save_index(self, path: str = None):
        if self.index and self._initialized:
            save_path = path or self.index_path or "knowledge_base.index"
            faiss.write_index(self.index, save_path)

    def load_index(self, path: str = None):
        load_path = path or self.index_path
        if load_path:
            self.index = faiss.read_index(load_path)
            self._initialized = True

    def remove_by_id(self, doc_id: int):
        if self.index and doc_id in self.custom_id_to_text:
            self.index.remove_ids(np.array([doc_id]).astype('int64'))
            del self.custom_id_to_text[doc_id]
            del self.custom_id_to_embedding[doc_id]

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[tuple]:
        if not self._initialized or self.index is None:
            return []
        query_emb = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_emb, k)
        results = []
        for dist, custom_id in zip(distances[0], indices[0]):
            if custom_id in self.custom_id_to_text:
                results.append((self.custom_id_to_text[custom_id], float(dist)))
        return results

    @property
    def size(self) -> int:
        return len(self.custom_id_to_text) if self._initialized else 0
