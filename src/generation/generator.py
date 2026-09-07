from transformers import pipeline
from src.models import (
    MinimalSource,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from tqdm import tqdm


class Generate:
    def __init__(self) -> None:
        self.generator = pipeline("text-generation",
                                  model="Qwen/Qwen2.5-0.5B-Instruct")
        self.instructions = (
            "You are a helpful assistant. Answer the question based on the "
            "provided sources. If the answer is not contained within the "
            "sources, respond with 'I don't know.'\n\n"
        )

    def _read_text_src(self, source: MinimalSource) -> str:
        try:
            with open(source.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {source.file_path}: {e}")
            return ""
        first_character_index = source.first_character_index
        last_character_index = source.last_character_index
        return content[first_character_index:last_character_index]

    def _build_prompt(self, question: str,
                      sources: list[MinimalSource]) -> str:
        blocks: list[str] = []
        for i, source in enumerate(sources, start=1):
            src_text = self._read_text_src(source)
            blocks.append(f"[{i}] {source.file_path}:\n{src_text}")
        context = "\n\n".join(blocks)
        prompt = (
            f"{self.instructions}"
            f"Question: {question}\n\nSources:\n{context}"
        )
        return prompt

    def generate(self, question: str, sources: list[MinimalSource]) -> str:
        prompt = self._build_prompt(question, sources)
        response = self.generator(prompt, max_new_tokens=256,
                                  num_return_sequences=1,
                                  return_full_text=False)
        return response[0]["generated_text"].strip()

    def generate_dataset(self, search_results: StudentSearchResults
                         ) -> StudentSearchResultsAndAnswer:
        list_of_answers: list[MinimalAnswer] = []
        for result in tqdm(search_results.search_results,
                           desc="Generating answers"):
            answer = self.generate(result.question,
                                   result.retrieved_sources)
            list_of_answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer,
                )
            )
        return StudentSearchResultsAndAnswer(
            search_results=list_of_answers,
            k=search_results.k,
        )
