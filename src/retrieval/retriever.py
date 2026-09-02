from pathlib import Path
from tqdm import tqdm
from src.models import (
    MinimalSource,
    StudentSearchResults,
    RagDataset,
    MinimalSearchResults,
)
from src.indexing.tokenize import tokenize
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from src.indexing.chunking import Chunk
import json


class Retrivial:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] | None = None

    def load_index(self) -> None:
        with open(self.index_dir / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self.index_dir / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, query: str, k: int) -> list[MinimalSource]:
        if self.bm25 is None or self.chunks is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_chunks = [self.chunks[i] for i in top_k_indices]
        return [
            MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            )
            for chunk in top_k_chunks
        ]

    def search_dataset(self, dataset_path: str,
                       k: int) -> StudentSearchResults:
        with open(dataset_path, "r") as f:
            data = json.load(f)
        dataset = RagDataset.model_validate(data)
        questions = dataset.rag_questions

        search_results: list[MinimalSearchResults] = []
        for question in tqdm(questions, desc="Searching dataset"):
            retrieved_sources = self.search(question.question, k)
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=retrieved_sources,
                )
            )
        return StudentSearchResults(search_results=search_results, k=k)
