import ast
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


if __name__ == "__main__":
    chunker = Chunking()
    chunks = chunker.chunk_py("src/models.py")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(chunk.text)
        print(f"First character index: {chunk.first_character_index}")
        print(f"Last character index: {chunk.last_character_index}")
        print("-" * 40)
