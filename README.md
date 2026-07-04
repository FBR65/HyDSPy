# HyDE-RAG System with DSPy and LanceDB

This project implements a Retrieval-Augmented Generation (RAG) system that combines Hypothetical Document Embeddings (HyDE) with DSPy. The system is modular and configurable, supporting any OpenAI-compatible endpoint.

## Features

- **HyDE (Hypothetical Document Embeddings)**: Improves semantic search by generating hypothetical documents
- **Multi-Query Reformulation**: The LLM generates up to N reformulations of the question, broadening vector-search coverage across different phrasings of the same intent
- **SIRA BM25 Hybrid Retrieval**: Query-side term prediction with a document-frequency (DF) filter and weighted Okapi-BM25, fused with the vector ranking via Reciprocal Rank Fusion (RRF). Closes lexical gaps that embeddings miss (e.g. error codes, identifiers, version numbers)
- **Agentic RAG Loop**: An iterative retrieval-reasoning loop that equips the LLM with `search`, `find`, `open`, and `summarize` tools. The agent decides when enough evidence has been gathered instead of finalising retrieval before reasoning begins
- **DSPy**: Optimises prompts and the full pipeline for better results
- **LanceDB**: Local vector database for efficient similarity search
- **Modular architecture**: Pluggable components for LLMs, embedding models, and databases
- **RESTful API**: FastAPI interface for document processing and question answering, with a per-request retrieval-mode switch
- **Tests**: Full pytest suite with no external dependencies

## Requirements

- Python >= 3.9
- [uv](https://github.com/astral-sh/uv) package manager
- Optional: a running OpenAI-compatible API (e.g. Ollama at `http://localhost:11434/v1`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd HyDE_DSPy
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv sync
   ```

3. Create a `.env` file from `.env.example` and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## Configuration

Configuration lives in `src/config/settings.yaml`. You can set different LLMs, embedding models, and the database here. Environment variables are referenced via `${VAR_NAME}`.

Options under `settings:`:

| Key | Default | Description |
|---|---|---|
| `chunk_size` | `1000` | Target size (characters) the LLM is asked to aim for when segmenting. The LLM chooses topic boundaries itself; this value is a guideline for the prompt and a hard upper bound on the final chunks. |
| `multi_query` | `true` | Enables Multi-Query Reformulation. The LLM generates reformulations of the question, each used for an additional vector search. |
| `num_queries` | `5` | Maximum number of reformulated queries produced per question. |
| `bm25_hybrid` | `true` | Enables the SIRA BM25 hybrid branch. Term prediction, DF filtering and weighted BM25 are combined with the vector ranking via RRF. |
| `expansion_weight` | `0.5` | Weight `w` in the weighted BM25 formula `BM25(q_orig) + w * BM25(q_exp)`. |
| `df_threshold` | `0.1` | Max fraction of documents a predicted term may occur in to survive the DF filter (`0 < DF <= df_threshold * N`). |
| `agentic_enabled` | `true` | Instantiates the `AgenticRAG` module at startup so the `agentic` retrieval mode is available via the API. |
| `max_calls` | `15` | Maximum number of iterations in the agentic loop before a final answer is forced. |
| `token_threshold` | `115200` | Token estimate (90% of 128k) at which the agentic loop forces a `summarize` call to compress the evidence trail. The token budget may be exceeded; the threshold only triggers summarisation. |
| `search_limit` | `10` | Maximum number of results returned per query inside the agentic `search` tool. |
| `max_queries_per_call` | `5` | Maximum number of reformulations accepted in a single agentic `search` tool call. |
| `open_window` | `5` | Number of chunks returned per `open` tool call (window-based reading). |

The `lancedb_config` block controls vector-index tuning and BM25 cache persistence:

| Key | Default | Description |
|---|---|---|
| `index_type` | `IVF_PQ` | Vector index type (`IVF_PQ`, `IVF_FLAT`, `IVF_HNSW_SQ`, `IVF_HNSW_PQ`). `null` disables the index build (sequential scan, suitable for small corpora or tests). |
| `num_partitions` | `null` | Number of IVF partitions. `null` lets LanceDB choose (default: sqrt(N)). |
| `num_sub_vectors` | `null` | Number of PQ sub-vectors. `null` lets LanceDB choose. Only relevant for `IVF_PQ`/`IVF_HNSW_PQ`. |
| `index_train_threshold` | `256` | Minimum row count before a vector index is built. Below this threshold LanceDB uses a sequential scan. |
| `bm25_cache_path` | `./lancedb/bm25_cache.pkl` | Path to a pickle file used to persist the BM25 index. `null` disables persistence (in-memory only). |

## Usage

### Command line

Run the main application (processes a demo document and asks a sample question):

```bash
uv run python src/main.py
```

### API

Start the API server:

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Or directly:

```bash
uv run python src/api/app.py
```

OpenAPI documentation is available at `http://localhost:8000/docs`.

#### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/process_document` | Process a document and store it in the database |
| `POST` | `/ask` | Answer a question based on the stored documents |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{id}` | Remove all chunks of a document |
| `GET` | `/health` | Check the API status |

#### Retrieval mode

The `/ask` endpoint accepts a `mode` field that selects the retrieval strategy per request:

| Mode | Description |
|---|---|
| `hyde` (default) | Single-shot HyDE with Multi-Query Reformulation and SIRA BM25 hybrid fusion. Fast, low token overhead. |
| `agentic` | Iterative Agentic RAG loop with `search`, `find`, `open`, and `summarize` tools. Higher recall and token cost; suited to multi-step questions. |

If the agentic pipeline is not initialised, the `agentic` mode returns `503`. Unknown mode values return `422`.

#### Example API usage

```bash
# Process a document
curl -X POST "http://localhost:8000/process_document" \
     -H "Content-Type: application/json" \
     -d '{"id": "doc1", "text": "Your document text here..."}'

# Ask a question in HyDE mode (optionally restrict to one document)
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here...", "document_id": "doc1", "k": 3, "mode": "hyde"}'

# Ask a question in agentic mode
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here...", "mode": "agentic"}'

# List documents
curl "http://localhost:8000/documents"

# Delete a document
curl -X DELETE "http://localhost:8000/documents/doc1"

# Health check
curl "http://localhost:8000/health"
```

Input limits (Pydantic validation):

- `Document.text`: max. 1,000,000 characters
- `Document.id`: 1-256 characters
- `Question.question`: 1-4,000 characters
- `Question.k`: 1-20 (default 3)
- `Question.document_id`: optional, max. 256 characters
- `Question.mode`: `hyde` or `agentic` (default `hyde`)

## Architecture

The data flow when processing and querying documents:

```
                    ┌───────────────────────┐
                    │  POST /process_document│
                    └───────────┬───────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │ SemanticChunker                 │
              │  • LLM picks topic boundaries   │
              │    AND chunk size based on      │
              │    semantic coherence           │
              │  • chunk_size is a guideline    │
              │    for the prompt + hard cap    │
              └──────────┬──────────────────────┘
                         │ List[str]
                         ▼
              ┌─────────────────────────────────┐
              │ EmbeddingManager                │
              │  • OpenAI-compatible /embeddings│
              │  • cached dimension             │
              └──────────┬──────────────────────┘
                         │ List[List[float]]
                         ▼
              ┌─────────────────────────────────┐
              │ DocumentProcessor               │
              │  • attaches metadata            │
              │  • writes to LanceDB            │
              └─────────────────────────────────┘
```

The query path (HyDE mode):

```
                    ┌───────────────────────┐
                    │  POST /ask            │
                    │  {question, doc_id?,  │
                    │   k, mode="hyde"}     │
                    └───────────┬───────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │ HyDERAG (dspy.Module)           │
              │  1. ChainOfThought ->           │
              │     hypothetical document       │
              │  2. Multi-Query Reformulation   │
              │     (up to num_queries)         │
              │  3. Vector search (HyDE + MQ)   │
              │     (optional: document_id)     │
              │  4. SIRA BM25 (term prediction, │
              │     DF filter, weighted BM25)   │
              │  5. RRF fusion (vector + BM25)  │
              │  6. ChainOfThought -> answer    │
              └─────────────────────────────────┘
```

The query path (agentic mode):

```
                    ┌───────────────────────┐
                    │  POST /ask            │
                    │  {question, doc_id?,  │
                    │   k, mode="agentic"}  │
                    └───────────┬───────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │ AgenticRAG (dspy.Module)        │
              │  for i = 1 to max_calls:        │
              │    • LLM reasons + emits either │
              │      a tool call or final answer│
              │    • tools: search / find /     │
              │      open / summarize           │
              │    • reference ids track        │
              │      retrieved evidence         │
              │    • token_threshold triggers   │
              │      summarize                  │
              │  no tool call -> return answer  │
              │  max_calls reached -> forced    │
              │  final answer                   │
              └─────────────────────────────────┘
```

Initialisation: `src/bootstrap.py:build()` is the central entry point used by both `main.py` and `src/api/app.py`. It loads configuration, instantiates all managers (including `BM25Manager` and `AgenticRAG`), and configures DSPy.

## Retrieval strategies

### Multi-Query Reformulation (HyDE mode)

Before retrieval, the LLM produces up to `num_queries` reformulations of the question. Each reformulation is embedded and used for a separate vector search. Results are deduplicated. This broadens coverage to documents that phrase the answer differently from the original question.

### SIRA BM25 Hybrid (HyDE mode)

Implements the query-compilation approach from "Before Another RAG Hop, Try Compiling the Query for BM25" (Akcam, 2026):

1. **Term prediction**: The LLM predicts terms likely to appear verbatim in the answer document but absent from the query (e.g. error codes, identifiers, product names).
2. **DF filter**: Predicted terms are filtered by document frequency. Terms absent from the corpus (`DF == 0`) or too common (`DF > 0.1 * N`) are dropped.
3. **Weighted BM25**: The combined score is `BM25(q_orig) + 0.5 * BM25(q_exp)`, where `q_orig` is the raw question and `q_exp` is the filtered expansion terms.
4. **RRF fusion**: The BM25 ranking and the vector ranking are fused via Reciprocal Rank Fusion (`k = 60`).

The BM25 index is built in memory from the LanceDB chunk texts and rebuilt lazily when the underlying table changes (detected via row count).

### Agentic RAG Loop (agentic mode)

Implements the iterative retrieval-reasoning loop from "AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases" (Microsoft, arXiv:2605.05538). The LLM is equipped with four coarse-to-fine tools:

| Tool | Input | Output | Purpose |
|---|---|---|---|
| `search` | list of queries (up to `max_queries_per_call`) | snippets with reference ids | corpus-wide search |
| `find` | reference id, keyword patterns | matching passages | in-document substring search |
| `open` | reference id, chunk position | window of chunk text | window-based document reading |
| `summarize` | none | compressed evidence trail | context management |

Loop control:

- Iterates up to `max_calls` (default 15) times.
- Terminates when the LLM produces a final answer without a tool call.
- A forced final answer is produced when `max_calls` is reached.
- Reference ids (`turnNsearchM`) allow `find` and `open` to reuse earlier results without re-querying.
- When the token estimate reaches `token_threshold`, a `summarize` call is forced to compress the evidence trail. The token budget is not a hard cap.

## Project structure

```
HyDSPy/
├── src/
│   ├── bootstrap.py              # Central component initialisation
│   ├── main.py                   # CLI demo
│   ├── api/
│   │   └── app.py                # FastAPI application (lifespan, endpoints, mode switch)
│   ├── components/
│   │   ├── agentic_rag.py        # Iterative Agentic RAG loop (dspy.Module)
│   │   ├── bm25_manager.py       # SIRA BM25 index, DF filter, weighted scoring
│   │   ├── chunker.py            # LLM-based SemanticChunker
│   │   ├── document_processor.py # Chunking + embedding + storage
│   │   ├── embedding_manager.py  # OpenAI-compatible embeddings
│   │   ├── hyde_rag.py           # HyDE + Multi-Query + BM25 hybrid (dspy.Module)
│   │   ├── lancedb_manager.py    # Vector database wrapper
│   │   └── llm_manager.py        # OpenAI-compatible LLM calls
│   ├── config/
│   │   ├── config_manager.py     # YAML + env-var substitution
│   │   └── settings.yaml         # Default configuration
│   └── utils/                    # (reserved for shared helpers)
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_api.py               # End-to-end API tests (incl. mode switch)
│   ├── test_bootstrap.py
│   ├── test_chunker.py
│   ├── test_config_manager.py
│   ├── test_embedding_manager.py
│   ├── test_enhancements.py        # Index tuning, BM25 cache, parser robustness
│   ├── test_hyde_rag.py
│   ├── test_lancedb_manager.py
│   └── test_llm_manager.py
├── .env.example
├── .flake8                       # Flake8 configuration (line-length 88, aligned with Black)
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

## Development

### Code quality

The project uses Black for formatting and Flake8 for linting. Both are configured to a line length of 88 (`pyproject.toml` for Black, `.flake8` for Flake8). Run them before committing:

```bash
uv run black .
uv run flake8 .
```

### Tests

Tests run completely without external services (LLMs are stubbed, LanceDB uses `tmp_path`):

```bash
uv run pytest
```

Verbose output:

```bash
uv run pytest -v
```

Run individual test modules:

```bash
uv run pytest tests/test_chunker.py
uv run pytest tests/test_api.py::test_ask_agentic_mode
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.