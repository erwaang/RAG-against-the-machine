"""Chunking strategies for Python and Markdown source files."""

import ast
import re
from src.models import MinimalSource


class Chunk(MinimalSource):
    """A source location together with the text it covers."""

    text: str


class Chunking:
    """Split files into chunks bounded by a maximum character size."""

    def __init__(self, chunk_size: int = 2000):
        """Store the maximum chunk size in characters.

        Args:
            chunk_size: Maximum number of characters per chunk.
        """
        self.chunk_size = chunk_size

    def _char_offsets(self, code: str) -> list[int]:
        """Compute the character offset at the start of each line.

        Args:
            code: The full text of a file.

        Returns:
            A list where index ``i`` is the character offset of line
            ``i`` (0-indexed), with a trailing entry for the end of file.
        """
        offsets = [0]
        for line in code.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return offsets

    def __split_into_chunks(self, text: str, start_index: int,
                            file_path: str,
                            boundaries: list[int] | None = None
                            ) -> list[Chunk]:
        """Hard-split a span of text into fixed-size chunks.

        Args:
            text: The text to split.
            start_index: Character offset of ``text`` within the file.
            file_path: Path of the file ``text`` was read from.
            boundaries: Character offsets within ``text`` that are safe
                to cut at (e.g. method starts). Each cut snaps back to
                the nearest boundary at or before the naive cut, so a
                split never falls in the middle of a method.

        Returns:
            A list of chunks, each at most ``self.chunk_size`` characters.
        """
        chunks = []
        i = 0
        while i < len(text):
            end = min(i + self.chunk_size, len(text))
            if boundaries:
                snapped = max((b for b in boundaries if i < b <= end),
                              default=None)
                if snapped is not None:
                    end = snapped
            chunk_text = text[i:end]
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start_index + i,
                    last_character_index=start_index + end,
                    text=chunk_text,
                )
            )
            i = end
        return chunks

    def chunk_py(self, file: str) -> list[Chunk]:
        """Chunk a Python file along its top-level AST nodes.

        Each import, function, class, and async function definition
        (with its decorators) becomes its own chunk, so a chunk always
        holds a self-contained unit of code. Oversized definitions are
        hard-split into smaller chunks.

        Args:
            file: Path to the Python file to chunk.

        Returns:
            The list of chunks extracted from the file (empty on read or
            parse errors).
        """
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
                decorators = getattr(node, "decorator_list", [])
                start_line = decorators[0].lineno if decorators else node.lineno
                end_lineno = node.end_lineno if node.end_lineno is not None \
                    else start_line
                start_char = offsets[start_line - 1]
                end_char = offsets[end_lineno]
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
                    boundaries = None
                    if isinstance(node, ast.ClassDef):
                        boundaries = [
                            offsets[(m.decorator_list[0].lineno
                                     if m.decorator_list else m.lineno) - 1]
                            - start_char
                            for m in node.body
                            if isinstance(m, (ast.FunctionDef,
                                              ast.AsyncFunctionDef))
                        ]
                    elif isinstance(node, (ast.FunctionDef,
                                           ast.AsyncFunctionDef)):
                        boundaries = [
                            offsets[stmt.lineno - 1] - start_char
                            for stmt in node.body
                        ]
                    chunks.extend(self.__split_into_chunks(
                        text, start_char, file, boundaries))
        return chunks

    def chunk_txt(self, file: str) -> list[Chunk]:
        """Chunk a plain text file into fixed-size chunks.

        Unlike ``chunk_md``, this does not treat lines starting with
        ``#`` as headings, so non-Markdown ``.txt`` files (e.g.
        ``CMakeLists.txt``, requirements files) aren't shredded into a
        chunk per comment line.

        Args:
            file: Path to the text file to chunk.

        Returns:
            The list of chunks extracted from the file (empty on read
            errors or an empty file).
        """
        try:
            with open(file, "r", encoding="utf-8") as f:
                code = f.read()
        except OSError as e:
            print(f"Error reading {file}: {e}")
            return []

        if not code.strip():
            return []
        return self.__split_into_chunks(code, 0, file)

    def chunk_md(self, file: str) -> list[Chunk]:
        """Chunk a Markdown file along its headings.

        Each heading (``#`` to ``######``) starts a new chunk that runs
        until the next heading, so a chunk corresponds to one
        documentation section. Any text before the first heading is kept
        as a preamble chunk, and oversized sections are hard-split.

        Args:
            file: Path to the Markdown file to chunk.

        Returns:
            The list of chunks extracted from the file (empty on read
            errors or an empty file).
        """
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
        """Add a Markdown section, hard-splitting it if it is too long."""
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
