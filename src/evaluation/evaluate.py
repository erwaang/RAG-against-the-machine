"""Recall@k computation between student search results and ground truth."""

import json
from pathlib import Path
from typing import Dict, List

from src.models import MinimalSource, RagDataset, StudentSearchResults


def _overlaps(a: MinimalSource, b: MinimalSource, min_iou: float = 0.05) -> bool:
    """Check whether two sources refer to overlapping regions of the same file.

    Args:
        a: A retrieved source.
        b: A ground-truth source.
        min_iou: Minimum intersection-over-union to count as a match.

    Returns:
        True if both sources are in the same file and their character
        ranges overlap enough (IoU >= min_iou).
    """
    if a.file_path != b.file_path:
        return False
    start = max(a.first_character_index, b.first_character_index)
    end = min(a.last_character_index, b.last_character_index)
    intersection = max(0, end - start)
    if intersection == 0:
        return False
    union_start = min(a.first_character_index, b.first_character_index)
    union_end = max(a.last_character_index, b.last_character_index)
    union = union_end - union_start
    if union <= 0:
        return False
    return (intersection / union) >= min_iou


class Evaluate:
    """Compute recall@k of student search results against ground truth."""

    def __init__(self, output_dir: Path = Path("data/output")) -> None:
        """Create the output directory used to persist evaluation reports.

        Args:
            output_dir: Directory to write ``evaluation_results.json`` into.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_reference_sources(self, dataset_path: Path) -> Dict[str, List[MinimalSource]]:
        """Load a ground-truth dataset into a question_id -> sources map.

        Args:
            dataset_path: Path to an AnsweredQuestions dataset JSON file.

        Returns:
            A mapping from question id to its ground-truth sources.
            Questions without a ``sources`` field are skipped.
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = RagDataset.model_validate(data)
        reference_sources: Dict[str, List[MinimalSource]] = {}
        for question in dataset.rag_questions:
            sources = getattr(question, "sources", None)
            if sources is not None:
                reference_sources[question.question_id] = sources
        return reference_sources

    def _load_student_results(
        self, student_search_results_path: Path
    ) -> StudentSearchResults:
        """Load a StudentSearchResults JSON file.

        Args:
            student_search_results_path: Path to the file produced by
                ``search_dataset``.

        Returns:
            The parsed StudentSearchResults.
        """
        with open(student_search_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return StudentSearchResults.model_validate(data)

    def recall_at_k(
        self,
        student_search_results_path: Path,
        dataset_path: Path,
        k_values: List[int] | None = None,
    ) -> Dict[int, float]:
        """Compute recall@k for each requested k.

        Args:
            student_search_results_path: Path to a StudentSearchResults JSON
                file produced by ``search_dataset``.
            dataset_path: Path to the ground-truth AnsweredQuestions dataset.
            k_values: Values of k to report. Defaults to [1, 3, 5, 10].

        Returns:
            A mapping from k to the average recall@k over all questions
            that have ground-truth sources.
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        student_results = self._load_student_results(student_search_results_path)
        reference_sources = self._load_reference_sources(dataset_path)

        recalls: Dict[int, List[float]] = {k: [] for k in k_values}

        for result in student_results.search_results:
            ref_sources = reference_sources.get(result.question_id)
            if not ref_sources:
                continue
            for k in k_values:
                retrieved = result.retrieved_sources[:k]
                found = 0
                for gt in ref_sources:
                    if any(_overlaps(r, gt) for r in retrieved):
                        found += 1
                recalls[k].append(found / len(ref_sources))

        return {
            k: (sum(values) / len(values) if values else 0.0)
            for k, values in recalls.items()
        }

    def evaluate(
        self,
        student_search_results_path: Path,
        dataset_path: Path,
        k_values: List[int] | None = None,
    ) -> Dict[int, float]:
        """Compute recall@k and persist a report under ``output_dir``."""
        recalls = self.recall_at_k(student_search_results_path, dataset_path, k_values)
        report = {f"recall@{k}": v for k, v in sorted(recalls.items())}
        with open(self.output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return recalls
