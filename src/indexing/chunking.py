import ast
import re
from src.models import MinimalSource


class Chunk(MinimalSource):
    text: str


class Chunking:
    def __init__(self, chunk_size: int = 2000):
        self.chunk_size = chunk_size

    def _char_offsets(self, code: str) -> list[int]:
        offsets = [0]
        for line in code.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return offsets

    def __split_into_chunks(self, text: str, start_index: int,
                            file_path: str) -> list[Chunk]:
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i:i + self.chunk_size]
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start_index + i,
                    last_character_index=start_index + i + len(chunk_text),
                    text=chunk_text,
                )
            )
        return chunks

    def chunk_py(self, file: str) -> list[Chunk]:
        try:
            with open(file, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading {file}: {e}")
            return []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"Error parsing {file}: {e}")
            return []

        offsets = self._char_offsets(code)
        chunks = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                 ast.ClassDef, ast.AsyncFunctionDef)):
                if getattr(node, "decorator_list", []):
                    start_line = node.decorator_list[0].lineno
                else:
                    start_line = node.lineno
                start_char = offsets[start_line - 1]
                end_char = offsets[node.end_lineno]
                text = code[start_char:end_char]
                if len(text) <= self.chunk_size:
                    chunks.append(
                        Chunk(
                            file_path=file,
                            first_character_index=start_char,
                            last_character_index=end_char,
                            text=text,
                        )
                    )
                else:
                    chunks.extend(self.__split_into_chunks(text, start_char,
                                                           file))
        return chunks

    def chunk_md(self, file: str) -> list[Chunk]:
        try:
            with open(file, "r", encoding="utf-8") as f:
                code = f.read()
        except OSError as e:
            print(f"Error reading {file}: {e}")
            return []

        offsets = self._char_offsets(code)
        lines = code.splitlines(keepends=True)

        header_pattern = re.compile(r"^#{1,6}\s+")
        header_line_indices = [
            i for i, line in enumerate(lines) if header_pattern.match(line)
        ]

        chunks = []

        if not header_line_indices:
            if code.strip():
                chunks.extend(self.__split_into_chunks(code, 0, file))
            return chunks

        if header_line_indices[0] > 0:
            start_char = 0
            end_char = offsets[header_line_indices[0]]
            preamble = code[start_char:end_char].strip()
            if preamble:
                self.__add_section(chunks, preamble, start_char, end_char,
                                    file)

        for idx, line_idx in enumerate(header_line_indices):
            start_char = offsets[line_idx]
            if idx + 1 < len(header_line_indices):
                end_line_idx = header_line_indices[idx + 1]
            else:
                end_line_idx = len(lines)
            end_char = offsets[end_line_idx]

            section_text = code[start_char:end_char].strip()
            if section_text:
                self.__add_section(chunks, section_text, start_char,
                                    end_char, file)

        return chunks

    def __add_section(self, chunks: list[Chunk], text: str, start_char: int,
                       end_char: int, file: str) -> None:
        """Ajoute une section Markdown, en la sous-découpant si trop longue."""
        if len(text) <= self.chunk_size:
            chunks.append(
                Chunk(
                    file_path=file,
                    first_character_index=start_char,
                    last_character_index=end_char,
                    text=text,
                )
            )
        else:
            chunks.extend(self.__split_into_chunks(text, start_char, file))


if __name__ == "__main__":
    chunker = Chunking()
    chunks = chunker.chunk_md("data/raw/vllm-0.10.1/README.md")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(chunk.text)
        print(f"First character index: {chunk.first_character_index}")
        print(f"Last character index: {chunk.last_character_index}")
        print("-" * 40)