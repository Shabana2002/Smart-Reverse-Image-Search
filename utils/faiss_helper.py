import faiss
import numpy as np


class FaissIndex:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, features):
        self.index.add(np.array(features).astype("float32"))

    def search(self, query, k=5):
        scores, indices = self.index.search(
            np.array([query]).astype("float32"),
            k
        )
        return scores, indices
