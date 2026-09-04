from src.retrieval.retriever import Retrivial
from pathlib import Path


class Evaluate:
    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = processed_dir
        self.retriever = Retrivial(processed_dir)
        self.retriever.load_index()
        self.output_dir = Path("data/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, dataset_path: Path, k: int = 5) -> None:
        results = self.retriever.search_dataset(dataset_path, k)
        with open(self.output_dir / "evaluation_results.json", "w") as f:
            f.write(results.model_dump_json())
