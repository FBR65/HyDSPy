"""Tests for src.components.agentic_rag.AgenticRAG and AgenticTools.

Covers the loop control logic that is not exercised by the parser tests in
test_enhancements.py:

- ``k`` passed to ``forward`` is forwarded to ``db_manager.search`` as ``limit``.
- reference ids are reset per turn (``turnNsearchM`` with M starting at 1 each
  turn).
- the loop terminates when the LLM produces a final answer (no tool call).
- ``max_calls`` forces a final answer when the LLM keeps emitting tool calls.
- ``summarize`` compresses the evidence trail.
- ``get_document_chunks`` (not private ``_open_table``) backs the ``open`` tool.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.components.agentic_rag import (
    AgenticRAG,
    AgenticState,
    AgenticTools,
    SearchResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.search.return_value = pd.DataFrame(
        {
            "text": ["alpha text", "beta text"],
            "document_id": ["d1", "d1"],
            "chunk_id": ["c1", "c2"],
            "_distance": [0.1, 0.2],
        }
    )
    return db


@pytest.fixture
def mock_embedding():
    em = MagicMock()
    em.get_embeddings.return_value = [[0.0] * 4]
    return em


@pytest.fixture
def agentic(mock_db, mock_embedding):
    rag = AgenticRAG(
        db_manager=mock_db,
        embedding_manager=mock_embedding,
        bm25_manager=None,
        max_calls=5,
    )
    # Stub the DSPy ChainOfThought step so no real LLM is contacted.
    rag.step = MagicMock()
    return rag


# ---------------------------------------------------------------------------
# k forwarding
# ---------------------------------------------------------------------------


def test_forward_forwards_k_as_search_limit(agentic, mock_db):
    """k passed to forward() must constrain db_manager.search(limit=...)."""
    # First call -> search tool call; second call -> final answer.
    agentic.step.side_effect = [
        MagicMock(next_action='{"tool": "search", "args": {"queries": ["q"]}}'),
        MagicMock(next_action="The answer is 42."),
    ]
    agentic("Question?", k=7)
    _, kwargs = mock_db.search.call_args
    assert kwargs["limit"] == 7


def test_forward_default_k_when_not_provided(agentic, mock_db):
    agentic.step.side_effect = [
        MagicMock(next_action='{"tool": "search", "args": {"queries": ["q"]}}'),
        MagicMock(next_action="done."),
    ]
    agentic("Question?")  # no k -> default_k=3
    _, kwargs = mock_db.search.call_args
    assert kwargs["limit"] == 3


# ---------------------------------------------------------------------------
# Reference counter reset per turn
# ---------------------------------------------------------------------------


def test_reference_counter_resets_per_turn():
    """Two search tool calls in different turns should both produce
    turn{N}search1 as the first reference id, proving the counter resets per
    turn rather than growing globally."""
    db = MagicMock()
    db.search.return_value = pd.DataFrame(
        {
            "text": ["one", "two"],
            "document_id": ["d1", "d1"],
            "chunk_id": ["c1", "c2"],
            "_distance": [0.1, 0.2],
        }
    )
    emb = MagicMock()
    emb.get_embeddings.return_value = [[0.0] * 4]
    tools = AgenticTools(db, emb)

    state = AgenticState(question="q")
    state.iteration = 1
    first = tools.search(state, ["q"])
    assert [r.reference_id for r in first] == ["turn1search1", "turn1search2"]

    state.iteration = 2
    second = tools.search(state, ["q"])
    assert [r.reference_id for r in second] == ["turn2search1", "turn2search2"]


def test_reference_counter_reset_between_searches_in_same_turn():
    """Two searches in the same turn should restart at search1 each, since the
    counter is reset at the start of every search call."""
    db = MagicMock()
    db.search.return_value = pd.DataFrame(
        {
            "text": ["one"],
            "document_id": ["d1"],
            "chunk_id": ["c1"],
            "_distance": [0.1],
        }
    )
    emb = MagicMock()
    emb.get_embeddings.return_value = [[0.0] * 4]
    tools = AgenticTools(db, emb)

    state = AgenticState(question="q")
    state.iteration = 3
    first = tools.search(state, ["q"])
    second = tools.search(state, ["q"])
    assert first[0].reference_id == "turn3search1"
    assert second[0].reference_id == "turn3search1"


# ---------------------------------------------------------------------------
# Loop termination
# ---------------------------------------------------------------------------


def test_loop_terminates_on_final_answer(agentic):
    """When the LLM emits a final answer (no tool call), the loop stops and
    returns that answer without further iterations."""
    agentic.step.side_effect = [
        MagicMock(next_action='{"tool": "search", "args": {"queries": ["q"]}}'),
        MagicMock(next_action="Final answer: 42"),
    ]
    result = agentic("Question?")
    assert "42" in result.answer
    # Exactly two step calls (one tool turn + one final-answer turn).
    assert agentic.step.call_count == 2


def test_loop_forces_final_answer_at_max_calls(agentic):
    """When max_calls is reached and the LLM keeps calling tools, a forced
    final answer is produced via the configured LM."""
    agentic.max_calls = 2
    # Every turn emits a search tool call -> never terminates voluntarily.
    agentic.step.side_effect = [
        MagicMock(next_action='{"tool": "search", "args": {"queries": ["q"]}}'),
        MagicMock(next_action='{"tool": "search", "args": {"queries": ["q"]}}'),
    ]
    # Stub _force_final_answer so it does not hit a real API.
    agentic._force_final_answer = lambda state, last_action: "forced answer"
    result = agentic("Question?")
    assert result.answer == "forced answer"
    assert agentic.step.call_count == 2


# ---------------------------------------------------------------------------
# Summarize tool
# ---------------------------------------------------------------------------


def test_summarize_truncates_stored_text():
    tools = AgenticTools(MagicMock(), MagicMock())
    state = AgenticState(question="q")
    long_text = "x" * 500
    state.references["turn1search1"] = SearchResult(
        reference_id="turn1search1",
        document_id="d1",
        chunk_id="c1",
        text=long_text,
        score=0.0,
    )
    summary = tools.summarize(state)
    assert "turn1search1" in summary
    # Stored text is truncated to ~200 chars + ellipsis.
    assert len(state.references["turn1search1"].text) <= 203
    assert state.references["turn1search1"].text.endswith("...")


def test_summarize_empty_references_returns_message():
    tools = AgenticTools(MagicMock(), MagicMock())
    state = AgenticState(question="q")
    assert "No evidence" in tools.summarize(state)


# ---------------------------------------------------------------------------
# open tool uses public get_document_chunks (not _open_table)
# ---------------------------------------------------------------------------


def test_open_uses_public_get_document_chunks():
    """open() must call db_manager.get_document_chunks (public API) rather than
    accessing _open_table internals."""
    db = MagicMock()
    db.get_document_chunks.return_value = pd.DataFrame(
        {
            "document_id": ["d1", "d1"],
            "chunk_index": [0, 1],
            "text": ["chunk zero", "chunk one"],
        }
    )
    tools = AgenticTools(db, MagicMock(), open_window=2)
    state = AgenticState(question="q")
    state.references["turn1search1"] = SearchResult(
        reference_id="turn1search1",
        document_id="d1",
        chunk_id="c1",
        text="chunk zero",
        score=0.0,
    )
    out = tools.open(state, "turn1search1", position=0)
    db.get_document_chunks.assert_called_once_with("d1")
    assert "chunk zero" in out
    assert "chunk one" in out
    # The private method must NOT be called.
    db._open_table.assert_not_called()


def test_open_unknown_reference_returns_message():
    tools = AgenticTools(MagicMock(), MagicMock())
    state = AgenticState(question="q")
    out = tools.open(state, "missing_ref", 0)
    assert "Unknown" in out


def test_open_empty_document_returns_message():
    db = MagicMock()
    db.get_document_chunks.return_value = pd.DataFrame()
    tools = AgenticTools(db, MagicMock())
    state = AgenticState(question="q")
    state.references["turn1search1"] = SearchResult(
        reference_id="turn1search1",
        document_id="d1",
        chunk_id="c1",
        text="x",
        score=0.0,
    )
    out = tools.open(state, "turn1search1")
    assert "No chunks" in out


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


def test_gather_context_deduplicates_references():
    rag = AgenticRAG(
        db_manager=MagicMock(), embedding_manager=MagicMock(), bm25_manager=None
    )
    state = AgenticState(question="q")
    state.references["a"] = SearchResult("a", "d1", "c1", "same text", 0.0)
    state.references["b"] = SearchResult("b", "d1", "c2", "same text", 0.0)
    state.references["c"] = SearchResult("c", "d1", "c3", "other text", 0.0)
    ctx = rag._gather_context(state)
    # "same text" appears once despite two references.
    assert ctx.count("same text") == 1
    assert "other text" in ctx


def test_gather_context_empty_returns_message():
    rag = AgenticRAG(
        db_manager=MagicMock(), embedding_manager=MagicMock(), bm25_manager=None
    )
    state = AgenticState(question="q")
    assert "No relevant" in rag._gather_context(state)


# ---------------------------------------------------------------------------
# Token estimate / summarize forcing
# ---------------------------------------------------------------------------


def test_estimate_tokens_grows_with_history():
    state = AgenticState(question="q")
    base = AgenticRAG._estimate_tokens(state)
    state.tool_history.append({"result": "a" * 400})
    assert AgenticRAG._estimate_tokens(state) > base


def test_token_threshold_forces_summarize(agentic):
    """When the token estimate crosses the threshold, a summarize call is
    forced mid-loop."""
    agentic.token_threshold = 1  # trivially crossed
    # Force the tool history to be non-trivial so summarize has content.
    agentic.step.side_effect = [
        MagicMock(next_action='{"tool": "search", "args": {"queries": ["q"]}}'),
        MagicMock(next_action="final answer after summarize"),
    ]
    # Spy on summarize.
    called = {"summarize": False}
    original_summarize = agentic.tools.summarize

    def _spy(state):
        called["summarize"] = True
        return original_summarize(state)

    agentic.tools.summarize = _spy
    agentic("Question?")
    assert called["summarize"] is True
