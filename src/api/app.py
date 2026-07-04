import asyncio
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.bootstrap import Components, build

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Limits für Eingabevalidierung
MAX_DOCUMENT_TEXT_LENGTH = 1_000_000  # 1 MB Text pro Dokument
MAX_QUESTION_LENGTH = 4_000
MAX_DOCUMENT_ID_LENGTH = 256
MAX_K = 20

# Erlaubte Zeichen für document_id: alphanumerisch, sowie _, -, :, . und /.
# Verhindert Filter-/SQL-Injection über den document_id-Filter, da diese IDs
# direkt in LanceDB-SQL-Filter interpoliert werden (siehe LanceDBManager).
_DOCUMENT_ID_PATTERN = r"^[A-Za-z0-9_\-:./]+$"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisiert Komponenten asynchron im Hintergrund und teardown beim Shutdown.

    Die schwere Initialisierung (Modell-Clients, DSPy-LM, LanceDB) laeuft in
    einem Hintergrund-Task, sodass der Server sofort startet und Anfragen
    annimmt. Bis die Initialisierung abgeschlossen ist, liefern die Endpoints
    503 (``_get_components`` prueft ``app.state.components``).
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
                logger.error("Komponenten unvollständig initialisiert.")
                app.state.components = None
                app.state.init_error = "incomplete"
                return
            app.state.components = comps
            logger.info("API-Komponenten erfolgreich initialisiert.")
        except Exception as e:
            logger.exception("API-Initialisierung fehlgeschlagen")
            app.state.init_error = str(e)

    app.state.init_task = asyncio.create_task(_run_init())
    try:
        yield
    finally:
        if app.state.init_task is not None and not app.state.init_task.done():
            app.state.init_task.cancel()
        logger.info("API wird heruntergefahren.")


app = FastAPI(
    title="HyDE-RAG API",
    description="API für das HyDE-RAG System",
    lifespan=lifespan,
)


def _get_components() -> Components:
    """Liefert die initialisierten Komponenten oder 503, falls die
    Initialisierung noch laeuft oder fehlgeschlagen ist."""
    comps: Components | None = getattr(app.state, "components", None)
    if comps is not None:
        return comps
    init_task = getattr(app.state, "init_task", None)
    if init_task is not None and not init_task.done():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API wird noch initialisiert (siehe Server-Logs)",
        )
    err = getattr(app.state, "init_error", None)
    detail = (
        f"API-Initialisierung fehlgeschlagen: {err}"
        if err and err != "incomplete"
        else "API nicht initialisiert (siehe Server-Logs)"
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
        description="Eindeutige Dokument-ID",
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DOCUMENT_TEXT_LENGTH,
        description="Dokumenttext (max. 1 MB)",
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not re.fullmatch(_DOCUMENT_ID_PATTERN, v):
            raise ValueError(
                "document_id darf nur alphanumerische Zeichen sowie "
                "_ - : . / enthalten"
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
        description="Optional: Suche auf ein bestimmtes Dokument einschränken",
    )

    @field_validator("document_id")
    @classmethod
    def _validate_document_id(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.fullmatch(_DOCUMENT_ID_PATTERN, v):
            raise ValueError(
                "document_id darf nur alphanumerische Zeichen sowie "
                "_ - : . / enthalten"
            )
        return v

    k: int = Field(
        default=3, ge=1, le=MAX_K, description="Anzahl der abgerufenen Kontexte"
    )
    mode: str = Field(
        default="hyde",
        description="Retrieval-Modus: 'hyde' (HyDE + Multi-Query + BM25-Hybrid) "
        "oder 'agentic' (iterativer Agentic-RAG-Loop).",
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
    """
    Verarbeitet ein Dokument durch Chunking, Vektorisierung und speichert es in der Datenbank.
    """
    comps = _get_components()
    try:
        chunks = comps.document_processor.process_document(document.id, document.text)
        return ProcessedDocument(document_id=document.id, chunks=len(chunks))
    except Exception as e:
        logger.exception("Fehler bei der Dokumentenverarbeitung")
        raise HTTPException(
            status_code=500, detail=f"Fehler bei der Dokumentenverarbeitung: {e}"
        )


@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """
    Beantwortet eine Frage basierend auf den in der Datenbank gespeicherten Dokumenten.

    Über ``mode`` wird der Retrieval-Modus gewählt:
      - ``hyde``: Single-Shot HyDE mit Multi-Query-Reformulation und BM25-Hybrid (Default).
      - ``agentic``: Iterativer Agentic-RAG-Loop (search/find/open/summarize).
    """
    comps = _get_components()
    mode = (question.mode or "hyde").lower()
    try:
        if mode == "agentic":
            if comps.agentic_rag is None:
                raise HTTPException(
                    status_code=503,
                    detail="Agentic-RAG-Modus ist nicht initialisiert.",
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
                detail=f"Unbekannter mode '{mode}': 'hyde' oder 'agentic' erwartet.",
            )
        return Answer(answer=result.answer, context=result.context)
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.exception("Fragebeantwortung fehlgeschlagen")
        raise HTTPException(
            status_code=502, detail=f"Fragebeantwortung fehlgeschlagen: {e}"
        )
    except Exception as e:
        logger.exception("Unerwarteter Fehler bei der Fragebeantwortung")
        raise HTTPException(
            status_code=500, detail=f"Fehler bei der Fragebeantwortung: {e}"
        )


@app.get("/documents", response_model=list[DocumentSummary])
async def list_documents():
    """
    Liefert eine Übersicht aller indizierten Dokumente.
    """
    comps = _get_components()
    try:
        return comps.db_manager.list_documents()
    except Exception as e:
        logger.exception("Fehler beim Auflisten der Dokumente")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Entfernt alle Chunks eines Dokuments aus der Datenbank.
    """
    if len(document_id) > MAX_DOCUMENT_ID_LENGTH or not re.fullmatch(
        _DOCUMENT_ID_PATTERN, document_id
    ):
        raise HTTPException(
            status_code=400,
            detail="document_id ungültig (nur alphanumerisch und _ - : . /)",
        )
    comps = _get_components()
    try:
        removed = comps.db_manager.delete_document(document_id)
    except Exception as e:
        logger.exception("Fehler beim Löschen des Dokuments")
        raise HTTPException(status_code=500, detail=str(e))
    return {"document_id": document_id, "removed": removed}


@app.get("/health")
async def health_check():
    """
    Prüft den Status der API.
    """
    comps = getattr(app.state, "components", None)
    return {
        "status": "healthy" if comps is not None else "degraded",
        "components_initialised": comps is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
