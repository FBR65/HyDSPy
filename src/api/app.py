import logging
import os
import sys
from contextlib import asynccontextmanager

# Füge das Projekt-Root zum Python-Pfad hinzu, damit `src.*`-Imports funktionieren,
# unabhängig davon, von wo aus das Skript aufgerufen wird.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.bootstrap import Components, build

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Limits für Eingabevalidierung
MAX_DOCUMENT_TEXT_LENGTH = 1_000_000  # 1 MB Text pro Dokument
MAX_QUESTION_LENGTH = 4_000
MAX_DOCUMENT_ID_LENGTH = 256
MAX_K = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisiert und teardown der Komponenten."""
    logger.info("Initialisiere API-Komponenten ...")
    try:
        comps = build()
    except Exception as e:
        logger.error("API-Initialisierung fehlgeschlagen: %s", e)
        app.state.components = None
        yield
        return
    if not comps.initialised:
        logger.error("Komponenten unvollständig initialisiert.")
        app.state.components = None
        yield
        return
    app.state.components = comps
    logger.info("API-Komponenten erfolgreich initialisiert.")
    try:
        yield
    finally:
        logger.info("API wird heruntergefahren.")


app = FastAPI(
    title="HyDE-RAG API",
    description="API für das HyDE-RAG System",
    lifespan=lifespan,
)


def _get_components() -> Components:
    """Liefert die initialisierten Komponenten oder 503, falls die Initialisierung fehlschlug."""
    comps: Components | None = getattr(app.state, "components", None)
    if comps is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API nicht initialisiert (siehe Server-Logs)",
        )
    return comps


class Document(BaseModel):
    id: str = Field(..., min_length=1, max_length=MAX_DOCUMENT_ID_LENGTH, description="Eindeutige Dokument-ID")
    text: str = Field(..., min_length=1, max_length=MAX_DOCUMENT_TEXT_LENGTH, description="Dokumenttext (max. 1 MB)")


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
    k: int = Field(default=3, ge=1, le=MAX_K, description="Anzahl der abgerufenen Kontexte")


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
        raise HTTPException(status_code=500, detail=f"Fehler bei der Dokumentenverarbeitung: {e}")


@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """
    Beantwortet eine Frage basierend auf den in der Datenbank gespeicherten Dokumenten.
    """
    comps = _get_components()
    try:
        result = comps.hyde_rag(
            question.question, document_id=question.document_id, k=question.k
        )
        return Answer(answer=result.answer, context=result.context)
    except RuntimeError as e:
        logger.exception("Fragebeantwortung fehlgeschlagen")
        raise HTTPException(status_code=502, detail=f"Fragebeantwortung fehlgeschlagen: {e}")
    except Exception as e:
        logger.exception("Unerwarteter Fehler bei der Fragebeantwortung")
        raise HTTPException(status_code=500, detail=f"Fehler bei der Fragebeantwortung: {e}")


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
    if len(document_id) > MAX_DOCUMENT_ID_LENGTH:
        raise HTTPException(status_code=400, detail="document_id zu lang")
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
