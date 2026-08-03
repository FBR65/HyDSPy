import asyncio
import logging
import re
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.bootstrap import Components, build

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Input validation limits
MAX_DOCUMENT_TEXT_LENGTH = 1_000_000  # 1 MB text per document
MAX_QUESTION_LENGTH = 4_000
MAX_DOCUMENT_ID_LENGTH = 256
MAX_K = 20

# Allowed characters for document_id: alphanumeric plus _, -, :, . and /.
# Prevents SQL/filter injection via the document_id filter, which is
# interpolated directly into LanceDB SQL-style filters (see LanceDBManager).
_DOCUMENT_ID_PATTERN = r"^[A-Za-z0-9_\-:./]+$"

# Retrieval modes accepted by the /ask endpoint.
RetrievalMode = Literal["hyde", "agentic"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise components asynchronously in the background and teardown on
    shutdown.

    The heavy initialisation (model clients, DSPy-LM, LanceDB) runs in a
    background task so the server starts and accepts requests immediately.
    Until initialisation is complete, endpoints return 503
    (``_get_components`` checks ``app.state.components``).
    """
    app.state.components = None
    app.state.init_task = None
    app.state.init_error = None

    def _init() -> Components:
        return build()

    async def _run_init():
        loop = asyncio.get_running_loop()
        try:
            comps = await loop.run_in_executor(None, _init)
            if not comps.initialised:
                logger.error("Components not fully initialised.")
                app.state.components = None
                app.state.init_error = "incomplete"
                return
            app.state.components = comps
            logger.info("API components initialised successfully.")
        except Exception as e:
            logger.exception("API initialisation failed")
            app.state.init_error = str(e)

    app.state.init_task = asyncio.create_task(_run_init())
    try:
        yield
    finally:
        if app.state.init_task is not None and not app.state.init_task.done():
            app.state.init_task.cancel()
        logger.info("API shutting down.")


app = FastAPI(
    title="HyDE-RAG API",
    description="API for the HyDE-RAG system",
    lifespan=lifespan,
)


def _get_components() -> Components:
    """Return initialised components or raise 503 if initialisation is still
    in progress or has failed."""
    comps: Components | None = getattr(app.state, "components", None)
    if comps is not None:
        return comps
    init_task = getattr(app.state, "init_task", None)
    if init_task is not None and not init_task.done():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API is still initialising (see server logs)",
        )
    err = getattr(app.state, "init_error", None)
    detail = (
        "API initialisation failed: see server logs"
        if err
        else "API not initialised (see server logs)"
    )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=detail,
    )


class Document(BaseModel):
    id: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DOCUMENT_ID_LENGTH,
        description="Unique document ID",
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DOCUMENT_TEXT_LENGTH,
        description="Document text (max. 1 MB)",
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not re.fullmatch(_DOCUMENT_ID_PATTERN, v):
            raise ValueError(
                "document_id may only contain alphanumeric characters and _ - : . /"
            )
        return v


class ProcessedDocument(BaseModel):
    document_id: str
    chunks: int


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    document_id: str | None = Field(
        default=None,
        max_length=MAX_DOCUMENT_ID_LENGTH,
        description="Optional: restrict search to a specific document",
    )

    @field_validator("document_id")
    @classmethod
    def _validate_document_id(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.fullmatch(_DOCUMENT_ID_PATTERN, v):
            raise ValueError(
                "document_id may only contain alphanumeric characters and _ - : . /"
            )
        return v

    k: int = Field(
        default=3, ge=1, le=MAX_K, description="Number of contexts retrieved"
    )
    mode: RetrievalMode = Field(
        default="hyde",
        description=(
            "Retrieval mode: 'hyde' (HyDE + Multi-Query + BM25-Hybrid) "
            "or 'agentic' (iterative Agentic-RAG loop)."
        ),
    )


class Answer(BaseModel):
    answer: str
    context: str


class DocumentSummary(BaseModel):
    document_id: str
    chunk_count: int
    first_chunk_index: int
    last_chunk_index: int


@app.post("/process_document", response_model=ProcessedDocument)
async def process_document(document: Document):
    """Process a document through chunking, vectorisation, and store it in the
    database."""
    comps = _get_components()
    try:
        chunks = comps.document_processor.process_document(document.id, document.text)
        return ProcessedDocument(document_id=document.id, chunks=len(chunks))
    except Exception:
        logger.exception("Document processing failed")
        raise HTTPException(
            status_code=500, detail="Document processing failed (see server logs)"
        )


@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """Answer a question based on documents stored in the database.

    The ``mode`` parameter selects the retrieval strategy:
      - ``hyde``: Single-shot HyDE with Multi-Query reformulation and BM25-Hybrid.
      - ``agentic``: Iterative Agentic-RAG loop (search/find/open/summarize).
    """
    comps = _get_components()
    mode = (question.mode or "hyde").lower()
    try:
        if mode == "agentic":
            if comps.agentic_rag is None:
                raise HTTPException(
                    status_code=503,
                    detail="Agentic-RAG mode is not initialised.",
                )
            result = comps.agentic_rag(
                question.question,
                document_id=question.document_id,
                k=question.k,
            )
        elif mode == "hyde":
            result = comps.hyde_rag(
                question.question,
                document_id=question.document_id,
                k=question.k,
            )
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown mode '{mode}': expected 'hyde' or 'agentic'.",
            )
        return Answer(answer=result.answer, context=result.context)
    except HTTPException:
        raise
    except RuntimeError:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=502, detail="Question answering failed (see server logs)"
        )
    except Exception:
        logger.exception("Unexpected error during question answering")
        raise HTTPException(
            status_code=500, detail="Question answering failed (see server logs)"
        )


@app.get("/documents", response_model=list[DocumentSummary])
async def list_documents():
    """Return an overview of all indexed documents."""
    comps = _get_components()
    try:
        return comps.db_manager.list_documents()
    except Exception:
        logger.exception("Failed to list documents")
        raise HTTPException(
            status_code=500, detail="Failed to list documents (see server logs)"
        )


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Remove all chunks belonging to a document from the database."""
    if len(document_id) > MAX_DOCUMENT_ID_LENGTH or not re.fullmatch(
        _DOCUMENT_ID_PATTERN, document_id
    ):
        raise HTTPException(
            status_code=400,
            detail="document_id invalid (only alphanumeric and _ - : . / allowed)",
        )
    comps = _get_components()
    try:
        removed = comps.db_manager.delete_document(document_id)
    except Exception:
        logger.exception("Failed to delete document")
        raise HTTPException(
            status_code=500, detail="Failed to delete document (see server logs)"
        )
    return {"document_id": document_id, "removed": removed}


@app.get("/health")
async def health_check():
    """Check the API status."""
    comps = getattr(app.state, "components", None)
    return {
        "status": "healthy" if comps is not None else "degraded",
        "components_initialised": comps is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
