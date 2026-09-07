"""Pydantic models exchanged between the indexing, retrieval, generation,
and evaluation stages of the RAG pipeline.
"""

from typing import List
from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """A single source location: a file and the character range it covers."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without a known answer, as read from a question dataset."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question together with its ground-truth sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or not, read from JSON."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources together with the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output of ``search_dataset``: retrieval results for a whole dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of ``answer_dataset``: retrieval results with answers."""

    search_results: List[MinimalAnswer]
    k: int
