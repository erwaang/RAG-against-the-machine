*This project has been created as part of the 42 curriculum by egaudich.*

# RAG against the machine

## Description

**RAG against the machine** is a Retrieval-Augmented Generation system that
answers natural-language questions about a real codebase (the
[vLLM](https://github.com/vllm-project/vllm) repository) instead of relying
on a language model's frozen training data.

The pipeline has four stages:

1. **Indexing** — chunk every Python and Markdown file of the corpus and
   build a searchable lexical index (BM25).
2. **Retrieval** — given a question, return the top-k most relevant chunks
   (file path + character range).
3. **Augmenting** — assemble the retrieved chunks into a bounded context
   window for the model.
4. **Generation** — produce a grounded, natural-language answer with
   `Qwen/Qwen3-0.6B`.

Retrieval quality is measured with recall@k: the share of the ground-truth
sources for a question that the system actually retrieves in its top-k
results.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) as package manager
- The vLLM repository extracted under `data/raw/` (not included in this
  repository — see the subject's attachments)

### Install

```bash
make install        # uv sync
```

### Run

Every command is exposed through `python -m src <command>`, and the
`Makefile` forwards extra arguments via `ARGS`:

```bash
# Build the index (defaults to --max_chunk_size 2000)
uv run python -m src index --max_chunk_size 2000

# Search a single question
uv run python -m src search "How do I configure the OpenAI server?" --k 5

# Search a whole dataset and save a StudentSearchResults file
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# Answer a single question end-to-end (retrieval + generation)
uv run python -m src answer "How do I configure the OpenAI server?" --k 5

# Generate answers for a whole search_dataset output
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# Evaluate recall@k against a ground-truth dataset (own testing)
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 5
```

Or via the Makefile:

```bash
make run ARGS="index --max_chunk_size 2000"
make debug ARGS="search 'my question' --k 5"
make lint
make clean
```

## System architecture

```
data/raw/  --(Indexer + Chunking)-->  data/processed/ (BM25 index + chunks)
                                              |
question --(Retrivial.search)----------------+--> top-k MinimalSource
                                              |
top-k sources --(Generate)-------------------+--> grounded answer (Qwen3-0.6B)
```

- `src/indexing/` — file discovery, chunking strategies, index building
  (`Indexer`, `Chunking`).
- `src/retrieval/` — loads the persisted index and ranks chunks against a
  query with BM25 (`Retrivial`).
- `src/generation/` — builds a bounded prompt from retrieved sources and
  generates an answer with `Qwen/Qwen3-0.6B` (`Generate`).
- `src/evaluation/` — computes recall@k between a `StudentSearchResults`
  file and a ground-truth `AnsweredQuestions` dataset (`Evaluate`).
- `src/models.py` — the pydantic models shared between all stages.
- `src/cli.py` — the Fire-based CLI wiring the stages together.

## Chunking strategy

Two distinct strategies are implemented, both capped at `--max_chunk_size`
(2000 characters by default):

- **Python (`chunk_py`)** — parses the file with `ast` and emits one chunk
  per top-level `import`, `class`, `def`/`async def`, keeping decorators
  attached to their definition. This keeps each chunk semantically
  self-contained (a whole function or class) instead of cutting through
  the middle of a definition. Oversized definitions are hard-split on
  `chunk_size` boundaries as a fallback.
- **Markdown (`chunk_md`)** — splits on headings (`#` to `######`), so each
  chunk corresponds to one documentation section, which matches how
  questions usually target one concept per section. Any oversized section
  is hard-split the same way as the Python fallback.

Smaller chunk sizes trade recall for context precision: a smaller
`--max_chunk_size` means more, more focused chunks (usually better recall
for narrow doc questions) but a proportionally larger index and more
chunks that need to be scanned to reconstruct a full function/section.

## Retrieval method

Retrieval uses **BM25** (`rank_bm25.BM25Okapi`) over a custom tokenizer
(`src/indexing/tokenize.py`) that:

- splits identifiers on `_` and `camelCase` boundaries, so a query for
  "api server" can match the identifier `api_server`,
- strips punctuation and lowercasing artifacts are handled by BM25 itself.

At query time, the query is tokenized the same way, scored against every
chunk with `get_scores`, and the top-k chunks (by descending score) are
returned as `MinimalSource` objects. Ties and empty/degenerate queries
(`k <= 0` or an empty/whitespace query) return an empty result list instead
of an arbitrary ranking.

## Performance analysis

Indexing the full vLLM corpus with `--max_chunk_size 2000` stays well
within the 5-minute budget, and retrieval throughput comfortably meets the
90-second budget for 200 questions since BM25 lookups are pure NumPy
vector operations against a pre-tokenized corpus.

Recall@5 on the reference datasets meets the required thresholds (80% on
docs questions, 50% on code questions); code questions are inherently
harder for a lexical method since a question rarely quotes an identifier
verbatim, while documentation prose overlaps much more directly with how
questions are phrased.

## Design decisions

- **BM25 over TF-IDF**: BM25's term-frequency saturation and length
  normalization behave better on a corpus with very uneven document/chunk
  lengths (a two-line import chunk vs. a 2000-character function).
- **AST-based Python chunking**: chunking on `ast` nodes instead of fixed
  character windows keeps each chunk a coherent unit (a full function or
  class), which is what both BM25 scoring and the generator benefit from.
- **`enable_thinking=False` for Qwen3**: Qwen3-0.6B is a reasoning model by
  default; disabling its `<think>` trace keeps answers fast and on-budget
  on CPU-only hardware, which matters for `answer_dataset` on a whole
  dataset.
- **Context budget in the generator**: sources are concatenated into the
  prompt until a character budget is reached rather than passing every
  retrieved source unconditionally, to stay within the model's context
  window regardless of `k`.

## Challenges faced

- **Matching questions against identifiers vs. prose**: a question like
  "how does the API server handle streaming?" needs to match both the
  Markdown documentation prose and the `api_server.py` identifiers. The
  camelCase/`snake_case`-aware tokenizer was added specifically to close
  that gap for code questions.
- **Keeping generation within the model's context window**: naively
  concatenating all `k` retrieved sources could exceed the moulinette's
  2000-character-per-source cap and the model's own context budget; the
  generator now truncates the assembled context instead of trusting `k`
  alone.
- **Graceful degradation**: the CLI has to survive empty queries, `k=0`,
  missing files, and malformed JSON without an unhandled traceback; each
  command now validates its inputs and reports a clear message instead of
  crashing.

## Example usage

```bash
$ uv run python -m src index --max_chunk_size 2000
Ingestion complete! Indices saved under data/processed

$ uv run python -m src search "How do I configure the OpenAI server?" --k 3
data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md [2000:3169]
data/raw/vllm-0.10.1/docs/configuration/serve_args.md [0:94]
data/raw/vllm-0.10.1/docs/design/arch_overview.md [1474:2157]

$ uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

$ uv run python -m src answer "How do I configure the OpenAI server?" --k 5
To configure the OpenAI server, you can use the `vllm serve` command with the
appropriate model and configuration. The OpenAI API server can be started
using the `vllm serve` command, which is described in the
[OpenAI-Compatible API Server](../serving/openai_compatible_server.md)
document.

$ uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
Loaded 100 questions
Saved student_search_results_and_answer to data/output/search_results_and_answer/UnansweredQuestions/dataset_docs_public.json

$ uv run python -m src evaluate \

    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 5
Evaluation Results
========================================
Recall@1: 0.540
Recall@3: 0.750
Recall@5: 0.810
```

## Resources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- [BM25: The Next Generation of Lexical Search](https://en.wikipedia.org/wiki/Okapi_BM25)
- [rank_bm25 documentation](https://github.com/dorianbrown/rank_bm25)
- [Python Fire documentation](https://github.com/google/python-fire)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [vLLM documentation](https://docs.vllm.ai/)

### AI usage

An AI assistant was used to:

- To answer some questions I had about the subject
- review the subject and identify gaps against the mandatory requirements
- draft this README.