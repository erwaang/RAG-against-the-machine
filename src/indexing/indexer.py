from pathlib import Path
import re
from tokenize import tokenize
from src.indexing.chunking import Chunk, Chunking
from rank_bm25 import BM25Okapi
from tqdm import tqdm # display progress bar for long-running operations


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

    def tokenize(self, text: str) -> list[str]:
        text = re.sub(r'_', ' ', text)
        text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.lower().split()

    def build_index(self, chunks: list[Chunk]) -> tuple[BM25Okapi, list[Chunk]]:
        if not chunks:
            raise ValueError("No chunks to index.")
        tokenized_corpus = [self.tokenize(chunk.text) for chunk in chunks]
        return BM25Okapi(tokenized_corpus), chunks


if __name__ == "__main__":
    indexer = Indexer()
    chunker = Chunking()
    chunks = indexer.chunk_files(chunker)
    print(f"Total chunks created: {len(chunks)}")
    print(chunks[:5])
