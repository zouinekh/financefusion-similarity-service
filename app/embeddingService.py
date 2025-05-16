from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def compare_texts(self, text1: str, text2: str) -> float:
        emb1 = self.embed_text(text1).reshape(1, -1)
        emb2 = self.embed_text(text2).reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
