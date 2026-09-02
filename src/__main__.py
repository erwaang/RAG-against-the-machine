from src.indexing.indexer import Indexer
from src.indexing.chunking import Chunking
from src.indexing.tokenize import tokenize
from rank_bm25 import BM25Okapi
from pathlib import Path
from src.retrieval.retriever import Retrivial

if __name__ == "__main__":
    indexer = Indexer()
    chunker = Chunking()
    bm25, chunks = indexer.build_index(chunker)
    indexer.save_index(bm25, chunks, Path("data/processed"))
    retriever = Retrivial(Path("data/processed"))
    retriever.load_index()
    results = retriever.search("How to configure OpenAI server?", k=5)
    for r in results:
        print(r.file_path, r.first_character_index, r.last_character_index)