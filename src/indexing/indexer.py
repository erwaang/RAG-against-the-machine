"""Corpus ingestion: file discovery, chunking, and BM25 index building."""

from pathlib import Path
from src.indexing.chunking import Chunk, Chunking
from src.indexing.tokenize import tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pickle


class Indexer:
    """Ingest a raw corpus and build a persisted BM25 index from it."""

    def __init__(self, raw_dir: Path = Path("data/raw")) -> None:
        """Store the root directory of the raw corpus.

        Args:
            raw_dir: Directory to recursively search for source files.
        """
        self.raw_dir = raw_dir

    def collect_files(self) -> list[Path]:
        """Find every indexable file under ``raw_dir``.

        Returns:
            All ``.py``, ``.md``, and ``.txt`` files found recursively.
        """
        # Search for files with the specified extensions in the raw_dir and its subdirectories.
        py_files = list(self.raw_dir.rglob("*.py"))
        md_files = list(self.raw_dir.rglob("*.md"))
        txt_files = list(self.raw_dir.rglob("*.txt"))
        return py_files + md_files + txt_files

    def chunk_files(self, chunker: Chunking) -> list[Chunk]:
        """Chunk every indexable file with the matching strategy.

        Args:
            chunker: The chunking strategy to apply.

        Returns:
            All chunks produced across the corpus. Files that fail to
            read are skipped with a warning instead of aborting the run.
        """
        files = self.collect_files()
        all_chunks: list[Chunk] = []
        for file in tqdm(files, desc="Chunking files"):
            try:
                if file.suffix == ".py":
                    chunks = chunker.chunk_py(str(file))
                elif file.suffix == ".md":
                    chunks = chunker.chunk_md(str(file))
                elif file.suffix == ".txt":
                    chunks = chunker.chunk_txt(str(file))
                else:
                    continue
                all_chunks.extend(chunks)
            except (UnicodeDecodeError, OSError) as e:
                print(f"Skipping {file}: {e}")
        return all_chunks

    def build_index(self, chunker: Chunking) -> tuple[BM25Okapi, list[Chunk]]:
        """Chunk the corpus and build a BM25 index over the chunks.

        Args:
            chunker: The chunking strategy to apply.

        Returns:
            The fitted BM25 index together with the chunks it was built
            from (chunks and index share the same ordering).
        """
        chunks = self.chunk_files(chunker)
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        # b = the length of a document penalizes its score
        bm25 = BM25Okapi(tokenized_corpus, b=0.45)
        return bm25, chunks

    def save_index(self, bm25: BM25Okapi, chunks: list[Chunk],
                   index_dir: Path) -> None:
        """Persist the BM25 index and its chunks to disk.

        Args:
            bm25: The fitted BM25 index.
            chunks: The chunks the index was built from.
            index_dir: Directory to write the index files into.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(bm25, f)
        with open(index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
