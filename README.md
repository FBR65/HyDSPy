# HyDE-RAG System with DSPy and LanceDB

This project implements a Retrieval-Augmented Generation (RAG) system that combines Hypothetical Document Embeddings (HyDE) with DSPy. The system is modular and configurable, supporting any OpenAI-compatible endpoint.

## Features

- **HyDE (Hypothetical Document Embeddings)**: Improves semantic search by generating hypothetical documents
- **DSPy**: Optimises prompts and the full pipeline for better results
- **LanceDB**: Local vector database for efficient similarity search
- **Modular architecture**: Pluggable components for LLMs, embedding models, and databases
- **RESTful API**: FastAPI interface for document processing and question answering
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

#### Example API usage

```bash
# Process a document
curl -X POST "http://localhost:8000/process_document" \
     -H "Content-Type: application/json" \
     -d '{"id": "doc1", "text": "Your document text here..."}'

# Ask a question (optionally restrict to one document)
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here...", "document_id": "doc1", "k": 3}'

# List documents
curl "http://localhost:8000/documents"

# Delete a document
curl -X DELETE "http://localhost:8000/documents/doc1"

# Health check
curl "http://localhost:8000/health"
```

Input limits (Pydantic validation):

- `Document.text`: max. 1,000,000 characters
- `Document.id`: 1–256 characters
- `Question.question`: 1–4,000 characters
- `Question.k`: 1–20 (default 3)
- `Question.document_id`: optional, max. 256 characters

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

The query path:

```
                    ┌───────────────────────┐
                    │  POST /ask            │
                    │  {question, doc_id?, k}│
                    └───────────┬───────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │ HyDERAG (dspy.Module)          │
              │  1. ChainOfThought ->           │
              │     hypothetical document      │
              │  2. Embedding + LanceDB search  │
              │     (optional: document_id)     │
              │  3. ChainOfThought -> answer    │
              └─────────────────────────────────┘
```

Initialisation: `src/bootstrap.py:build()` is the central entry point used by both `main.py` and `src/api/app.py`. It loads configuration, instantiates all managers, and configures DSPy.

## Project structure

```
HyDSPy/
├── src/
│   ├── bootstrap.py              # Central component initialisation
│   ├── main.py                   # CLI demo
│   ├── api/
│   │   └── app.py                # FastAPI application (lifespan, endpoints)
│   ├── components/
│   │   ├── chunker.py            # LLM-based SemanticChunker
│   │   ├── document_processor.py # Chunking + embedding + storage
│   │   ├── embedding_manager.py  # OpenAI-compatible embeddings
│   │   ├── hyde_rag.py           # HyDE pipeline (dspy.Module)
│   │   ├── lancedb_manager.py    # Vector database wrapper
│   │   └── llm_manager.py        # OpenAI-compatible LLM calls
│   ├── config/
│   │   ├── config_manager.py     # YAML + env-var substitution
│   │   └── settings.yaml         # Default configuration
│   └── utils/                    # (reserved for shared helpers)
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_api.py               # End-to-end API tests
│   ├── test_bootstrap.py
│   ├── test_chunker.py
│   ├── test_config_manager.py
│   ├── test_embedding_manager.py
│   ├── test_hyde_rag.py
│   ├── test_lancedb_manager.py
│   └── test_llm_manager.py
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

## Development

### Code quality

The project uses Black for formatting and Flake8 for linting. Run them before committing:

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
uv run pytest tests/test_api.py::test_ask_with_document_id_filter_excludes_other_documents
```

## Known limitations

- **LLM-based chunking** requires a model that reliably returns JSON of the form `{"chunks": [...]}`. Bad responses raise `ValueError`. Pass `llm_manager=None` to `SemanticChunker` to disable the LLM path and fall back to a deterministic size-based splitter.
- **No authentication**: the API is open. A production deployment needs an auth layer (API key, OAuth, etc.).
- **No index tuning**: LanceDB tables use default indices. For >10k chunks configure `IVF_PQ` or `HNSW`.
- **`document_id` filter** is a SQL string interpolation. If document IDs ever become user-controlled, the value must be escaped.
- **Synchronous FastAPI lifespan**: initialisation blocks server startup. For slow model initialisation this should be moved to a background task.
- **No multi-tenancy**: all requests share the same table. Per-tenant isolation is not provided.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
