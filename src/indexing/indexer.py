from pathlib import Path
from src.indexing.chunking import Chunk, Chunking


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
        for file in files:
            if file.suffix == ".py":
                chunks = chunker.chunk_py(str(file))
            elif file.suffix == ".md":
                chunks = chunker.chunk_md(str(file))
            else:
                continue
            all_chunks.extend(chunks)
        return all_chunks
