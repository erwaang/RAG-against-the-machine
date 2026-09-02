from src.indexing.indexer import Indexer
from src.indexing.chunking import Chunking
from src.indexing.tokenize import tokenize
from rank_bm25 import BM25Okapi
from pathlib import Path


if __name__ == "__main__":
    indexer = Indexer()
    chunker = Chunking()
    bm25, chunks = indexer.build_index(chunker)
    indexer.save_index(bm25, chunks, Path("data/processed"))
    print(f"Indexed {len(chunks)} chunks.")