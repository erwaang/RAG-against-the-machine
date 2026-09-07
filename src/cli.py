"""Fire-based CLI wiring the indexing, retrieval, generation, and
evaluation stages together.
"""

import json
from pathlib import Path

from src.evaluation.evaluate import Evaluate
from src.generation.generator import Generate
from src.indexing.chunking import Chunking
from src.indexing.indexer import Indexer
from src.models import StudentSearchResults, StudentSearchResultsAndAnswer
from src.retrieval.retriever import Retrivial


class CLI:
    """Command-line interface for the RAG pipeline."""

    def __init__(self) -> None:
        """Set the default raw and processed data directories."""
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")

    def _load_retriever(self) -> Retrivial | None:
        """Load the persisted retriever, or None if no index exists yet."""
        retriever = Retrivial(self.processed_dir)
        try:
            retriever.load_index()
        except FileNotFoundError:
            print(
                f"No index found under {self.processed_dir}. "
                "Run `index` first."
            )
            return None
        return retriever

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/.

        Args:
            max_chunk_size: Maximum number of characters per chunk.
        """
        if not self.raw_dir.exists():
            print(f"Raw data directory {self.raw_dir} does not exist.")
            return
        try:
            indexer = Indexer(raw_dir=self.raw_dir)
            chunker = Chunking(chunk_size=max_chunk_size)
            bm25, chunks = indexer.build_index(chunker)
            if not chunks:
                print(f"No indexable files found under {self.raw_dir}.")
                return
            indexer.save_index(bm25, chunks, self.processed_dir)
            print(f"Ingestion complete! Indices saved under {self.processed_dir}")
        except Exception as e:
            print(f"Indexing failed: {e}")

    def search(self, query: str, k: int = 5) -> None:
        """Return the top-k sources for a single query.

        Args:
            query: The natural-language question to search for.
            k: Number of top results to return.
        """
        retriever = self._load_retriever()
        if retriever is None:
            return
        try:
            results = retriever.search(query, k)
        except Exception as e:
            print(f"Search failed: {e}")
            return
        if not results:
            print("No results.")
            return
        for r in results:
            print(f"{r.file_path} [{r.first_character_index}:{r.last_character_index}]")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Run search over a whole dataset and write a StudentSearchResults file.

        Args:
            dataset_path: Path to the UnansweredQuestions/AnsweredQuestions
                dataset JSON to search.
            k: Number of top results to return per question.
            save_directory: Directory to write the output file into; the
                output file keeps the dataset's original name.
        """
        retriever = self._load_retriever()
        if retriever is None:
            return
        try:
            results = retriever.search_dataset(dataset_path, k)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not read dataset {dataset_path}: {e}")
            return
        except Exception as e:
            print(f"search_dataset failed: {e}")
            return

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(dataset_path).name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(results.model_dump_json(indent=2))
        print(f"Saved student_search_results to {out_path}")

    def answer(self, query: str, k: int = 5) -> None:
        """Answer a single query using the retrieved context.

        Args:
            query: The natural-language question to answer.
            k: Number of top sources to retrieve and pass as context.
        """
        retriever = self._load_retriever()
        if retriever is None:
            return
        try:
            sources = retriever.search(query, k)
            generator = Generate()
            answer_text = generator.generate(query, sources)
        except Exception as e:
            print(f"answer failed: {e}")
            return
        print(answer_text)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a dataset of already-searched questions.

        Args:
            student_search_results_path: Path to a StudentSearchResults
                JSON file produced by ``search_dataset``.
            save_directory: Directory to write the output file into; the
                output file keeps the input file's original name.
        """
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            search_results = StudentSearchResults.model_validate(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not read {student_search_results_path}: {e}")
            return
        except Exception as e:
            print(f"Invalid student search results file: {e}")
            return

        print(f"Loaded {len(search_results.search_results)} questions")
        try:
            generator = Generate()
            results_with_answers: StudentSearchResultsAndAnswer = (
                generator.generate_dataset(search_results)
            )
        except Exception as e:
            print(f"answer_dataset failed: {e}")
            return

        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / Path(student_search_results_path).name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(results_with_answers.model_dump_json(indent=2))
        print(f"Saved student_search_results_and_answer to {out_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 5,
    ) -> None:
        """Report recall@k against a ground-truth dataset (own testing).

        Args:
            student_search_results_path: Path to a StudentSearchResults
                JSON file produced by ``search_dataset``.
            dataset_path: Path to the ground-truth AnsweredQuestions
                dataset JSON.
            k: An additional k value to report alongside 1, 3, and 5.
        """
        try:
            evaluator = Evaluate()
            recalls = evaluator.evaluate(
                Path(student_search_results_path),
                Path(dataset_path),
                k_values=sorted({1, 3, 5, k}),
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not read input files: {e}")
            return
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return

        print("Evaluation Results")
        print("=" * 40)
        for value_k, recall in sorted(recalls.items()):
            print(f"Recall@{value_k}: {recall:.3f}")
