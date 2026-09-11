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
            A list of the character offsets at the start of each line.
        """
        offsets = [0]
        # Add offsets for each line, including the trailing newline if present.
        for line in code.splitlines(keepends=True):
            # The offset of the next line is the offset of the current line + its length.
            offsets.append(offsets[-1] + len(line))
        return offsets

    def __split_into_chunks(self, text: str, start_index: int,
                            file_path: str,
                            safe_offsets: list[int] | None = None
                            ) -> list[Chunk]:
        """Hard-split a span of text into fixed-size chunks.

        Args:
            text: The text to split.
            start_index: Character offset of ``text`` within the file.
            file_path: Path of the file ``text`` was read from.
            safe_offsets: Character offsets within ``text`` that are safe
                to cut at (e.g. method starts). Each cut snaps back to
                the nearest safe offset at or before the naive cut, so a
                split never falls in the middle of a method.

        Returns:
            A list of chunks, each at most ``self.chunk_size`` characters.
        """
        chunks = []
        i = 0
        while i < len(text):
            # Prevent the chunk from extending beyond what actually exists.
            end = min(i + self.chunk_size, len(text))
            if safe_offsets:
                # Find the closest safe offset before the end of the chunk.
                cut_position = max((b for b in safe_offsets if i < b <= end),
                                   default=None)
                if cut_position is not None:
                    end = cut_position
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
            # AST = Abstract Syntax Tree
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
                # node.lineno = the line number where this code fragment start in the source file.
                start_line = decorators[0].lineno if decorators else node.lineno
                # node.end_lineno = the line number where this fragment ends in the source file.
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
                    safe_offsets = None
                    if isinstance(node, ast.ClassDef):
                        # Cut at the start of each method, including decorators
                        safe_offsets = [
                            offsets[(m.decorator_list[0].lineno
                                     if m.decorator_list else m.lineno) - 1]
                            - start_char
                            for m in node.body
                            if isinstance(m, (ast.FunctionDef,
                                              ast.AsyncFunctionDef))
                        ]
                    elif isinstance(node, (ast.FunctionDef,
                                           ast.AsyncFunctionDef)):
                        # Cut at the start of each statement in the function
                        safe_offsets = [
                            offsets[stmt.lineno - 1] - start_char
                            for stmt in node.body
                        ]
                    chunks.extend(self.__split_into_chunks(
                        text, start_char, file, safe_offsets))
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

        # strip = remove spaces/line breaks at the start and end of the text
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
        # Find the line indices of all Markdown headings in the file.
        header_line_indices = [
            i for i, line in enumerate(lines) if header_pattern.match(line)
        ]

        chunks = []

        if not header_line_indices:
            if code.strip():
                chunks.extend(self.__split_into_chunks(code, 0, file))
            return chunks

        # If there is text before the first heading, treat it as a preamble chunk.
        if header_line_indices[0] > 0:
            start_char = 0
            end_char = offsets[header_line_indices[0]]
            preamble = code[start_char:end_char].strip()
            if preamble:
                self.__add_section(chunks, preamble, start_char, end_char,
                                   file)

        # Split the text into sections based on the headings, and add each section as a chunk.
        # Example: If the first heading is at line 3 (0, 3)
        for index, line_index in enumerate(header_line_indices):
            start_char = offsets[line_index]
            # Determine the section end: start of next heading, or end of file.
            if index + 1 < len(header_line_indices):
                end_line_index = header_line_indices[index + 1]
            else:
                end_line_index = len(lines)
            end_char = offsets[end_line_index]

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
            # Hard-split it into smaller chunks.
            chunks.extend(self.__split_into_chunks(text, start_char, file))
