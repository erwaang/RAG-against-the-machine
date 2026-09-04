from pathlib import Path
from src.indexing.indexer import Indexer
from src.indexing.chunking import Chunking
from src.retrieval.retriever import Retrivial


class CLI:
    def __init__(self) -> None:
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")

    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(raw_dir=self.raw_dir)
        chunker = Chunking(chunk_size=max_chunk_size)
        bm25, chunks = indexer.build_index(chunker)
        indexer.save_index(bm25, chunks, self.processed_dir)
        print(f"Indexing complete. Index saved to {self.processed_dir}")
        try:
            retriever = Retrivial(self.processed_dir)
            retriever.load_index()
            print("Index loaded successfully.")
        except Exception as e:
            print(f"Error loading index: {e}")

    def search(self, query: str, k: int = 5) -> None:
        retriever = Retrivial(self.processed_dir)
        retriever.load_index()
        results = retriever.search(query, k)
        for r in results:
            print(r.file_path, r.first_character_index, r.last_character_index)
