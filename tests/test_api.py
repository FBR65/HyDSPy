"""End-to-end tests for the FastAPI application.

No real Ollama / LLM API is required. Components are replaced with stub
classes (fallback LLMs, tempdir LanceDB).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.bootstrap import Components
from src.components.agentic_rag import AgenticRAG
from src.components.bm25_manager import BM25Manager
from src.components.chunker import SemanticChunker
from src.components.document_processor import DocumentProcessor
from src.components.embedding_manager import EmbeddingManager
from src.components.hyde_rag import HyDERAG
from src.components.lancedb_manager import LanceDBManager
from src.config.config_manager import ConfigManager


class _StubEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _StubEmbeddingsResponse:
    def __init__(self, vectors):
        self.data = [_StubEmbeddingItem(v) for v in vectors]


class _StubEmbeddingsNamespace:
    def __init__(self, dim):
        self._dim = dim
        self.calls: list[dict] = []

    def create(self, *, input, model):
        self.calls.append({"input": input, "model": model})
        # Deterministic 8-dim vectors: 0.1, 0.2, 0.3, ... per input position.
        return _StubEmbeddingsResponse(
            [[0.1 * (i + 1)] * self._dim for i in range(len(input))]
        )


class _StubClient:
    def __init__(self, dim=8):
        self.embeddings = _StubEmbeddingsNamespace(dim)


@pytest.fixture
def stub_components(tmp_path, monkeypatch) -> Components:
    """Builds a fully functional Components instance without external services.

    The embedding client is monkey-patched to return deterministic 8-dim
    vectors so the LanceDB search path produces reproducible results.
    """
    cfg = ConfigManager()
    db_manager = LanceDBManager(uri=str(tmp_path), table_name="docs")
    bm25_manager = BM25Manager(db_manager=db_manager)
    comps = Components(
        config_manager=cfg,
        chunk_llm_manager=None,
        answer_llm_manager=None,
        embedding_manager=EmbeddingManager("http://localhost:1", "k", "fake"),
        db_manager=db_manager,
        chunker=SemanticChunker(llm_manager=None, chunk_size=500),
        document_processor=None,
        hyde_rag=None,
        bm25_manager=bm25_manager,
        agentic_rag=None,
    )
    # Replace the embedding client with a deterministic stub.
    monkeypatch.setattr(comps.embedding_manager, "client", _StubClient(dim=8))

    comps.document_processor = DocumentProcessor(
        comps.chunker, comps.embedding_manager, comps.db_manager
    )
    comps.hyde_rag = HyDERAG(
        comps.db_manager, comps.embedding_manager, bm25_manager=bm25_manager
    )
    comps.agentic_rag = AgenticRAG(
        comps.db_manager, comps.embedding_manager, bm25_manager=bm25_manager
    )
    # Stub the DSPy sub-modules so /ask returns a deterministic answer without
    # requiring a configured LM.
    comps.hyde_rag.generate_hyde = MagicMock(
        return_value=MagicMock(hypothetical_document="hypo")
    )
    comps.hyde_rag.generate_queries = MagicMock(return_value=MagicMock(queries=""))
    comps.hyde_rag.predict_terms = MagicMock(return_value=MagicMock(terms=""))
    comps.hyde_rag.final_rag = MagicMock(return_value=MagicMock(answer="stub answer"))
    # Stub the agentic pipeline so /ask?mode=agentic returns a deterministic
    # answer without running the real agentic loop.
    comps.agentic_rag = MagicMock(
        return_value=MagicMock(answer="agentic answer", context="agentic ctx")
    )
    return comps


@pytest.fixture
def stub_components_no_agentic(tmp_path, monkeypatch) -> Components:
    """Components with agentic_rag=None (agentic_enabled=False scenario)."""
    cfg = ConfigManager()
    db_manager = LanceDBManager(uri=str(tmp_path), table_name="docs")
    bm25_manager = BM25Manager(db_manager=db_manager)
    comps = Components(
        config_manager=cfg,
        chunk_llm_manager=None,
        answer_llm_manager=None,
        embedding_manager=EmbeddingManager("http://localhost:1", "k", "fake"),
        db_manager=db_manager,
        chunker=SemanticChunker(llm_manager=None, chunk_size=500),
        document_processor=None,
        hyde_rag=None,
        bm25_manager=bm25_manager,
        agentic_rag=None,
    )
    monkeypatch.setattr(comps.embedding_manager, "client", _StubClient(dim=8))

    comps.document_processor = DocumentProcessor(
        comps.chunker, comps.embedding_manager, comps.db_manager
    )
    comps.hyde_rag = HyDERAG(
        comps.db_manager, comps.embedding_manager, bm25_manager=bm25_manager
    )
    comps.hyde_rag.generate_hyde = MagicMock(
        return_value=MagicMock(hypothetical_document="hypo")
    )
    comps.hyde_rag.generate_queries = MagicMock(return_value=MagicMock(queries=""))
    comps.hyde_rag.predict_terms = MagicMock(return_value=MagicMock(terms=""))
    comps.hyde_rag.final_rag = MagicMock(return_value=MagicMock(answer="stub answer"))
    # agentic_rag stays None — simulating agentic_enabled=False.
    return comps


@pytest.fixture
def client(stub_components) -> TestClient:
    """TestClient with stub components via lifespan override."""

    @asynccontextmanager
    async def stub_lifespan(app):
        app.state.components = stub_components
        yield

    original = app_module.app.router.lifespan_context
    app_module.app.router.lifespan_context = stub_lifespan
    try:
        with TestClient(app_module.app) as c:
            yield c
    finally:
        app_module.app.router.lifespan_context = original


@pytest.fixture
def no_agentic_client(stub_components_no_agentic) -> TestClient:
    """TestClient with agentic_rag=None."""

    @asynccontextmanager
    async def stub_lifespan(app):
        app.state.components = stub_components_no_agentic
        yield

    original = app_module.app.router.lifespan_context
    app_module.app.router.lifespan_context = stub_lifespan
    try:
        with TestClient(app_module.app) as c:
            yield c
    finally:
        app_module.app.router.lifespan_context = original


@pytest.fixture
def broken_client() -> TestClient:
    """TestClient where initialisation fails."""

    @asynccontextmanager
    async def broken_lifespan(app):
        app.state.components = None
        yield

    original = app_module.app.router.lifespan_context
    app_module.app.router.lifespan_context = broken_lifespan
    try:
        with TestClient(app_module.app) as c:
            yield c
    finally:
        app_module.app.router.lifespan_context = original


# --- Route discovery ---


def test_required_routes_are_registered():
    paths = {r.path for r in app_module.app.routes if hasattr(r, "path")}
    assert {"/process_document", "/ask", "/health", "/documents"}.issubset(paths)
    assert "/documents/{document_id}" in paths


# --- /health ---


def test_health_healthy(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "components_initialised": True}


def test_health_degraded_when_uninitialised(broken_client):
    r = broken_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "degraded", "components_initialised": False}


# --- 503 when not initialised ---


@pytest.mark.parametrize(
    "method,url,json_body",
    [
        ("post", "/process_document", {"id": "x", "text": "y"}),
        ("post", "/ask", {"question": "x"}),
        ("get", "/documents", None),
        ("delete", "/documents/x", None),
    ],
)
def test_endpoints_return_503_when_uninitialised(broken_client, method, url, json_body):
    fn = getattr(broken_client, method)
    if json_body is not None:
        r = fn(url, json=json_body)
    else:
        r = fn(url)
    assert r.status_code == 503, f"{method} {url} -> {r.status_code} {r.text}"


# --- Pydantic validation ---


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},  # empty question
        {"question": "x", "k": 0},  # k too small
        {"question": "x", "k": 999},  # k too large
        {"question": "x", "document_id": "a" * 300},  # document_id too long
    ],
)
def test_ask_validation_422(client, payload):
    r = client.post("/ask", json=payload)
    assert r.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "", "text": "x"},  # empty id
        {"id": "abc"},  # text missing
        {"id": "a" * 300, "text": "x"},  # id too long
        {"id": "x", "text": ""},  # empty text
    ],
)
def test_process_document_validation_422(client, payload):
    r = client.post("/process_document", json=payload)
    assert r.status_code == 422


def test_ask_unknown_mode_returns_422(client):
    """With Literal mode, an unknown mode is rejected by Pydantic at
    validation time (422), not at runtime."""
    r = client.post("/ask", json={"question": "?", "mode": "bogus"})
    assert r.status_code == 422


# --- Happy path ---


def test_process_document_returns_chunk_count(client):
    r = client.post(
        "/process_document",
        json={"id": "doc1", "text": "Paragraph one.\n\nParagraph two."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["document_id"] == "doc1"
    assert body["chunks"] >= 1


def test_documents_lists_indexed_documents(client):
    client.post("/process_document", json={"id": "doc1", "text": "alpha"})
    client.post("/process_document", json={"id": "doc2", "text": "beta beta"})
    r = client.get("/documents")
    assert r.status_code == 200
    docs = r.json()
    ids = {d["document_id"] for d in docs}
    assert ids == {"doc1", "doc2"}


def test_ask_with_document_id_filter_excludes_other_documents(client):
    client.post(
        "/process_document",
        json={"id": "alpha", "text": "Content alpha. More alpha."},
    )
    client.post(
        "/process_document",
        json={"id": "beta", "text": "Different topic entirely."},
    )
    r = client.post("/ask", json={"question": "?", "document_id": "alpha", "k": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "alpha" in body["context"]
    assert "beta" not in body["context"]


def test_ask_default_parameters(client):
    client.post("/process_document", json={"id": "d", "text": "some text"})
    r = client.post("/ask", json={"question": "?"})
    assert r.status_code == 200
    assert "answer" in r.json()
    assert "context" in r.json()


def test_delete_document_removes_chunks(client):
    client.post("/process_document", json={"id": "doomed", "text": "text"})
    r = client.delete("/documents/doomed")
    assert r.status_code == 200
    assert r.json()["document_id"] == "doomed"
    assert r.json()["removed"] >= 1

    docs = client.get("/documents").json()
    assert "doomed" not in {d["document_id"] for d in docs}


def test_delete_nonexistent_document_returns_zero(client):
    r = client.delete("/documents/does-not-exist")
    assert r.status_code == 200
    assert r.json()["removed"] == 0


def test_delete_document_id_too_long(client):
    r = client.delete("/documents/" + "a" * 300)
    assert r.status_code == 400


# --- Input size limit on question ---


def test_ask_too_long_question_422(client):
    r = client.post("/ask", json={"question": "x" * 5000})
    assert r.status_code == 422


# --- mode switch (HyDE vs. Agentic) ---


def test_ask_default_mode_is_hyde(client):
    client.post("/process_document", json={"id": "d", "text": "some text"})
    r = client.post("/ask", json={"question": "?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "stub answer"


def test_ask_explicit_hyde_mode(client):
    client.post("/process_document", json={"id": "d", "text": "some text"})
    r = client.post("/ask", json={"question": "?", "mode": "hyde"})
    assert r.status_code == 200
    assert r.json()["answer"] == "stub answer"


def test_ask_agentic_mode(client):
    client.post("/process_document", json={"id": "d", "text": "some text"})
    r = client.post("/ask", json={"question": "?", "mode": "agentic"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "agentic answer"
    assert body["context"] == "agentic ctx"


def test_ask_agentic_mode_when_not_initialised(broken_client):
    r = broken_client.post("/ask", json={"question": "?", "mode": "agentic"})
    assert r.status_code == 503


def test_ask_agentic_mode_when_disabled(no_agentic_client):
    """When agentic_enabled=False (agentic_rag=None), /ask?mode=agentic
    must return 503 instead of crashing."""
    no_agentic_client.post("/process_document", json={"id": "d", "text": "some text"})
    r = no_agentic_client.post("/ask", json={"question": "?", "mode": "agentic"})
    assert r.status_code == 503
    assert "not initialised" in r.json()["detail"]
    assert r.json()["detail"] == "Agentic-RAG mode is not initialised."


def test_ask_hyde_mode_still_works_when_agentic_disabled(no_agentic_client):
    """HyDE mode must work even when agentic is disabled."""
    no_agentic_client.post("/process_document", json={"id": "d", "text": "some text"})
    r = no_agentic_client.post("/ask", json={"question": "?", "mode": "hyde"})
    assert r.status_code == 200
    assert r.json()["answer"] == "stub answer"


def test_ask_hides_internal_error_details(client):
    """When an unexpected exception is raised, the response must not leak
    exception details to the client (security)."""
    from unittest.mock import MagicMock as _MM

    # Force hyde_rag to raise an unexpected error.
    comps = app_module.app.state.components
    comps.hyde_rag.final_rag = _MM(
        side_effect=RuntimeError("secret internal path /etc/passwd")
    )
    client.post("/process_document", json={"id": "d", "text": "some text"})
    r = client.post("/ask", json={"question": "?"})
    # RuntimeError maps to 502; the detail must not contain the secret.
    assert r.status_code == 502
    assert "secret internal" not in r.json()["detail"]
