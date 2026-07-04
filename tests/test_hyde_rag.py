"""Tests for src.components.hyde_rag.HyDERAG.

Verifies that document_id and k are correctly forwarded to db_manager.search.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.components.hyde_rag import HyDERAG


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_embedding():
    em = MagicMock()
    em.get_embeddings.return_value = [[0.0] * 4]
    return em


def test_forward_passes_document_id_and_k_to_search(mock_db, mock_embedding):
    mock_db.search.return_value = pd.DataFrame(
        {"text": ["alpha", "beta"], "document_id": ["d1", "d1"]}
    )
    rag = HyDERAG(db_manager=mock_db, embedding_manager=mock_embedding)
    rag.generate_hyde = MagicMock(return_value=MagicMock(hypothetical_document="hypo"))
    rag.final_rag = MagicMock(return_value=MagicMock(answer="the answer"))

    result = rag("Question?", document_id="d1", k=2)

    mock_embedding.get_embeddings.assert_called_once_with(["hypo"])
    mock_db.search.assert_called_once_with([0.0] * 4, limit=2, document_id="d1")
    assert "alpha" in result.context
    assert "beta" in result.context
    assert result.answer == "the answer"


def test_forward_no_filter(mock_db, mock_embedding):
    mock_db.search.return_value = pd.DataFrame({"text": ["x"]})
    rag = HyDERAG(db_manager=mock_db, embedding_manager=mock_embedding)
    rag.generate_hyde = MagicMock(return_value=MagicMock(hypothetical_document="h"))
    rag.final_rag = MagicMock(return_value=MagicMock(answer="a"))

    rag("q?")

    _, kwargs = mock_db.search.call_args
    assert kwargs.get("document_id") is None


def test_forward_empty_search_returns_message(mock_db, mock_embedding):
    mock_db.search.return_value = pd.DataFrame()
    rag = HyDERAG(db_manager=mock_db, embedding_manager=mock_embedding)
    rag.generate_hyde = MagicMock(return_value=MagicMock(hypothetical_document="h"))
    rag.final_rag = MagicMock(return_value=MagicMock(answer="a"))

    result = rag("q?")
    assert result.context == "No relevant contexts found."


def test_forward_deduplicates_contexts(mock_db, mock_embedding):
    mock_db.search.return_value = pd.DataFrame({"text": ["a", "b", "a", "b", "c"]})
    rag = HyDERAG(db_manager=mock_db, embedding_manager=mock_embedding)
    rag.generate_hyde = MagicMock(return_value=MagicMock(hypothetical_document="h"))
    rag.final_rag = MagicMock(return_value=MagicMock(answer="x"))

    result = rag("q?")
    parts = result.context.split("\n\n")
    assert parts == ["a", "b", "c"]


def test_dspy_errors_propagate(mock_db, mock_embedding):
    """DSPy errors should propagate as-is (no fallback)."""
    rag = HyDERAG(db_manager=mock_db, embedding_manager=mock_embedding)
    rag.generate_hyde = MagicMock(side_effect=Exception("DSPy broken"))

    with pytest.raises(Exception, match="DSPy broken"):
        rag("q?")
