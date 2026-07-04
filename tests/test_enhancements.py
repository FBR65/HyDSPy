"""Tests fuer die Erweiterungen: Index-Tuning, BM25-Cache und robuster
Tool-Call-Parser im Agentic-Loop."""

from __future__ import annotations

import os
import pickle
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.components.bm25_manager import BM25Manager
from src.components.agentic_rag import AgenticRAG, AgenticTools, AgenticState
from src.components.lancedb_manager import LanceDBManager

# ---------------------------------------------------------------------------
# Index-Tuning (LanceDBManager)
# ---------------------------------------------------------------------------


def _make_docs(document_id: str, n_chunks: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "document_id": document_id,
                "chunk_id": f"{document_id}_chunk_{i}",
                "text": f"Inhalt von {document_id} chunk {i}",
                "vector": [float(i)] * 8,
                "source": "test",
                "chunk_index": i,
                "total_chunks": n_chunks,
            }
            for i in range(n_chunks)
        ]
    )


def test_index_disabled_when_type_is_none(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="t", index_type=None)
    mgr.add_documents_from_df(_make_docs("doc1", 300))
    assert mgr._index_built is False


def test_index_not_built_below_threshold(temp_lancedb_uri):
    mgr = LanceDBManager(
        uri=temp_lancedb_uri,
        table_name="t",
        index_type="IVF_FLAT",
        index_train_threshold=256,
    )
    mgr.add_documents_from_df(_make_docs("doc1", 10))
    assert mgr._index_built is False


def test_index_built_above_threshold(temp_lancedb_uri):
    mgr = LanceDBManager(
        uri=temp_lancedb_uri,
        table_name="t",
        index_type="IVF_FLAT",
        index_train_threshold=10,
    )
    mgr.add_documents_from_df(_make_docs("doc1", 20))
    assert mgr._index_built is True


def test_rebuild_index_forces_rebuild(temp_lancedb_uri):
    mgr = LanceDBManager(
        uri=temp_lancedb_uri,
        table_name="t",
        index_type="IVF_FLAT",
        index_train_threshold=10,
    )
    mgr.add_documents_from_df(_make_docs("doc1", 20))
    assert mgr._index_built is True
    mgr._index_built = False
    mgr.rebuild_index()
    assert mgr._index_built is True


# ---------------------------------------------------------------------------
# BM25-Cache-Persistenz
# ---------------------------------------------------------------------------


@pytest.fixture
def bm25_with_cache(tmp_path):
    uri = str(tmp_path)
    cache_path = os.path.join(str(tmp_path), "bm25_cache.pkl")
    db = LanceDBManager(uri=uri, table_name="t")
    return db, BM25Manager(db_manager=db, cache_path=cache_path), cache_path


def test_bm25_cache_persisted_after_refresh(bm25_with_cache):
    db, bm25, cache_path = bm25_with_cache
    db.add_documents_from_df(_make_docs("doc1", 3))
    bm25._refresh_index()
    assert os.path.exists(cache_path)
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    assert data["n_docs"] == 3
    assert data["row_count"] == 3


def test_bm25_cache_loaded_on_new_manager(bm25_with_cache):
    db, bm25, cache_path = bm25_with_cache
    db.add_documents_from_df(_make_docs("doc1", 3))
    bm25._refresh_index()
    # New manager reusing the same cache_path + db.
    bm25_2 = BM25Manager(db_manager=db, cache_path=cache_path)
    bm25_2._refresh_index()
    assert bm25_2._n_docs == 3
    assert bm25_2._bm25 is not None


def test_bm25_cache_invalidated_on_row_count_change(bm25_with_cache):
    db, bm25, cache_path = bm25_with_cache
    db.add_documents_from_df(_make_docs("doc1", 3))
    bm25._refresh_index()
    assert bm25._n_docs == 3
    db.add_documents_from_df(_make_docs("doc2", 2))
    bm25._refresh_index()
    assert bm25._n_docs == 5


def test_bm25_invalidate_forces_rebuild(bm25_with_cache):
    db, bm25, cache_path = bm25_with_cache
    db.add_documents_from_df(_make_docs("doc1", 3))
    bm25._refresh_index()
    bm25.invalidate()
    assert bm25._bm25 is None
    bm25._refresh_index()
    assert bm25._bm25 is not None


def test_bm25_no_cache_when_path_none(temp_lancedb_uri):
    db = LanceDBManager(uri=temp_lancedb_uri, table_name="t")
    db.add_documents_from_df(_make_docs("doc1", 2))
    bm25 = BM25Manager(db_manager=db, cache_path=None)
    bm25._refresh_index()
    assert bm25._n_docs == 2


# ---------------------------------------------------------------------------
# Robuster Tool-Call-Parser (AgenticRAG._parse_tool_call)
# ---------------------------------------------------------------------------


@pytest.fixture
def agentic():
    return AgenticRAG(
        db_manager=MagicMock(),
        embedding_manager=MagicMock(),
        bm25_manager=None,
    )


def test_parse_inline_json_tool_call(agentic):
    action = (
        'Based on the question I will {"tool": "search", "args": {"queries": ["q1"]}}'
    )
    result = agentic._parse_tool_call(action)
    assert result == {"tool": "search", "args": {"queries": ["q1"]}}


def test_parse_fenced_json_tool_call(agentic):
    action = (
        '```json\n{"tool": "find", "args": '
        '{"reference_id": "r1", "patterns": ["x"]}}\n```'
    )
    result = agentic._parse_tool_call(action)
    assert result["tool"] == "find"
    assert result["args"]["reference_id"] == "r1"


def test_parse_xml_tool_tag(agentic):
    action = '<tool name="open">{"reference_id": "r1", "position": 2}</tool>'
    result = agentic._parse_tool_call(action)
    assert result["tool"] == "open"
    assert result["args"]["reference_id"] == "r1"


def test_parse_heuristic_search_line(agentic):
    action = 'I should search: "query one"; "query two"'
    result = agentic._parse_tool_call(action)
    assert result is not None
    assert result["tool"] == "search"
    assert len(result["args"]["queries"]) == 2


def test_parse_final_answer_returns_none(agentic):
    action = "The answer is 42, based on the evidence."
    assert agentic._parse_tool_call(action) is None


def test_parse_empty_returns_none(agentic):
    assert agentic._parse_tool_call("") is None


def test_coerce_args_search_from_text(agentic):
    args = AgenticRAG._coerce_args_from_text("search", "alpha; beta")
    assert args == {"queries": ["alpha", "beta"]}


def test_coerce_args_find_from_text(agentic):
    args = AgenticRAG._coerce_args_from_text("find", "r1; termA; termB")
    assert args["reference_id"] == "r1"
    assert args["patterns"] == ["termA", "termB"]


def test_coerce_args_open_with_position(agentic):
    args = AgenticRAG._coerce_args_from_text("open", "r1; 3")
    assert args["reference_id"] == "r1"
    assert args["position"] == 3


# ---------------------------------------------------------------------------
# Agentic-Tools: search/find/open Smoke-Tests mit Mocks
# ---------------------------------------------------------------------------


def test_agentic_search_assigns_reference_ids():
    db = MagicMock()
    db.search.return_value = pd.DataFrame(
        {
            "text": ["alpha", "beta"],
            "document_id": ["d1", "d1"],
            "chunk_id": ["c1", "c2"],
        }
    )
    emb = MagicMock()
    emb.get_embeddings.return_value = [[0.1] * 8]
    tools = AgenticTools(db, emb)
    state = AgenticState(question="q")
    results = tools.search(state, ["q"])
    assert len(results) == 2
    assert results[0].reference_id.startswith("turn0search")
    assert state.references[results[0].reference_id].text == "alpha"


def test_agentic_find_unknown_reference_returns_message():
    tools = AgenticTools(MagicMock(), MagicMock())
    state = AgenticState(question="q")
    passages = tools.find(state, "unknown_ref", ["x"])
    assert "Unknown" in passages[0]


def test_agentic_open_unknown_reference_returns_message():
    tools = AgenticTools(MagicMock(), MagicMock())
    state = AgenticState(question="q")
    out = tools.open(state, "unknown_ref", 0)
    assert "Unknown" in out


# ---------------------------------------------------------------------------
# Async-Lifespan (API-Initialisierung im Hintergrund)
# ---------------------------------------------------------------------------


def test_lifespan_starts_background_init():
    """The lifespan should start an asyncio task rather than blocking."""
    import asyncio
    from contextlib import asynccontextmanager
    from src.api import app as app_module

    state = MagicMock()
    state.components = None
    state.init_task = None
    state.init_error = None

    @asynccontextmanager
    async def fake_lifespan(app):
        app.state.components = None
        app.state.init_task = None
        app.state.init_error = None

        async def _run():
            await asyncio.sleep(0.01)
            app.state.components = "done"

        app.state.init_task = asyncio.create_task(_run())
        yield

    original = app_module.app.router.lifespan_context
    app_module.app.router.lifespan_context = fake_lifespan
    try:
        from fastapi.testclient import TestClient

        with TestClient(app_module.app) as client:
            # Allow background task to finish.
            import time

            time.sleep(0.05)
            assert app_module.app.state.components == "done"
            r = client.get("/health")
            assert r.status_code == 200
    finally:
        app_module.app.router.lifespan_context = original
