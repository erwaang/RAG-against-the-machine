from pathlib import Path
from src.indexing.chunking import Chunk, Chunking
from src.indexing.tokenize import tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm # display progress bar for long-running operations
import pickle


class Indexer:
    def __init__(self, raw_dir: Path = Path("data/raw")) -> None:
        self.raw_dir = raw_dir

    def collect_files(self) -> list[Path]:
        py_files = list(self.raw_dir.rglob("*.py"))
        md_files = list(self.raw_dir.rglob("*.md"))
        return py_files + md_files

    def chunk_files(self, chunker: Chunking) -> list[Chunk]:
        files = self.collect_files()
        all_chunks: list[Chunk] = []
        for file in tqdm(files, desc="Chunking files"):
            try:
                if file.suffix == ".py":
                    chunks = chunker.chunk_py(str(file))
                elif file.suffix == ".md":
                    chunks = chunker.chunk_md(str(file))
                else:
                    continue
                all_chunks.extend(chunks)
            except (UnicodeDecodeError, OSError) as e:
                print(f"Skipping {file}: {e}")
        return all_chunks

    def build_index(self, chunker: Chunking) -> tuple[BM25Okapi, list[Chunk]]:
        chunks = self.chunk_files(chunker)
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, chunks

    def save_index(self, bm25: BM25Okapi, chunks: list[Chunk],
                   index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)   # Save the BM25 index to a file
        with open(index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)   # Save the list of chunks to a file
