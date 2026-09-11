"""Grounded answer generation with a small local causal LM (Qwen3-0.6B)."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models import (
    MinimalSource,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from tqdm import tqdm

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARS = 6000


class Generate:
    """Generate grounded answers from retrieved sources with a small LLM."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Load the tokenizer and causal LM used to generate answers.

        Runs on CUDA with float16 weights when a GPU is available,
        falling back to CPU float32 otherwise.

        Args:
            model_name: Hugging Face model id to load.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model = model.to(self.device)  # type: ignore[arg-type]
        self.instructions = (
            "You are a helpful assistant answering questions about a "
            "codebase. Answer only using the provided sources. If the "
            "answer is not contained within the sources, say so instead "
            "of guessing."
        )

    def _read_text_src(self, source: MinimalSource) -> str:
        """Read the text span a source refers to from disk.

        Args:
            source: The source location to read.

        Returns:
            The text between the source's character indices, or an empty
            string if the file cannot be read.
        """
        try:
            with open(source.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            print(f"Error reading {source.file_path}: {e}")
            return ""
        return content[source.first_character_index:source.last_character_index]

    def _build_prompt(self, question: str, sources: list[MinimalSource]) -> str:
        """Assemble a bounded prompt from the question and its sources.

        Sources are appended in order until ``MAX_CONTEXT_CHARS`` would be
        exceeded, so the prompt stays within the model's context budget
        regardless of how many sources are passed in.

        Args:
            question: The question being answered.
            sources: The retrieved sources to ground the answer in.

        Returns:
            The formatted prompt text (question plus numbered sources).
        """
        blocks: list[str] = []
        total_len = 0
        for i, source in enumerate(sources, start=1):
            src_text = self._read_text_src(source)
            if not src_text:
                continue
            block = f"[{i}] {source.file_path}:\n{src_text}"
            if total_len + len(block) > MAX_CONTEXT_CHARS:
                break
            blocks.append(block)
            total_len += len(block)
        context = "\n\n".join(blocks)
        return f"Question: {question}\n\nSources:\n{context}"

    def generate(self, question: str, sources: list[MinimalSource]) -> str:
        """Generate a grounded answer for a question from its sources.

        Args:
            question: The question to answer.
            sources: The retrieved sources to ground the answer in.

        Returns:
            The generated answer, or "I don't know." for an empty
            question or if generation fails.
        """
        if not question or not question.strip():
            return "I don't know."
        prompt = self._build_prompt(question, sources)
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            decoded = self.tokenizer.decode(
                generated, skip_special_tokens=True
            )
            answer = str(decoded).strip()
            return answer if answer else "I don't know."
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I don't know."

    def generate_dataset(
        self, search_results: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        """Generate an answer for every question in a search results set.

        Args:
            search_results: Retrieval results produced by
                ``Retrivial.search_dataset``.

        Returns:
            The same results with a generated answer attached to each
            question.
        """
        list_of_answers: list[MinimalAnswer] = []
        for result in tqdm(search_results.search_results, desc="Generating answers"):
            answer = self.generate(result.question, result.retrieved_sources)
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
