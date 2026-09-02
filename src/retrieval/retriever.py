from pathlib import Path
from src.models import MinimalSource
from src.indexing.tokenize import tokenize
import numpy as np
import pickle


class Retrivial:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir

    def search(self, query: str, k: int) -> list[MinimalSource]:
        tokenized_query = tokenize(query)
        bm25_path = self.index_dir / "bm25_index.pkl"
        chunks_path = self.index_dir / "chunks.pkl"
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        scores = bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        # Sort the indices by their scores in descending order
        # and keep the top k
        top_k_chunks = [chunks[i] for i in top_k_indices]
        # Get the top k chunks based on the scores
        return [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            )
            for chunk in top_k_chunks
        ]
