"""Tests for src.components.document_processor.DocumentProcessor."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.components.document_processor import DocumentProcessor


@pytest.fixture
def mock_chunker():
    chunker = MagicMock()
    chunker.chunk.return_value = ["chunk one", "chunk two"]
    return chunker


@pytest.fixture
def mock_embedding():
    emb = MagicMock()
    emb.get_embeddings.return_value = [[0.1] * 8, [0.2] * 8]
    return emb


def test_process_document_stores_chunks(mock_chunker, mock_embedding):
    db = MagicMock()
    db._open_table.return_value = MagicMock()
    proc = DocumentProcessor(mock_chunker, mock_embedding, db)
    result = proc.process_document("doc1", "some text")
    assert len(result) == 2
    assert result[0]["document_id"] == "doc1"
    assert result[0]["chunk_id"] == "doc1_chunk_0"
    assert result[1]["chunk_index"] == 1
    df_arg = db.add_documents_from_df.call_args[0][0]
    assert isinstance(df_arg, pd.DataFrame)
    assert len(df_arg) == 2


def test_process_document_empty_chunks_returns_empty(mock_chunker, mock_embedding):
    mock_chunker.chunk.return_value = []
    db = MagicMock()
    proc = DocumentProcessor(mock_chunker, mock_embedding, db)
    result = proc.process_document("doc1", "")
    assert result == []
    db.add_documents_from_df.assert_not_called()


def test_process_document_swallows_delete_when_table_missing(
    mock_chunker, mock_embedding
):
    """When delete_document raises and the table doesn't exist, the error
    should be swallowed (first-run scenario)."""
    db = MagicMock()
    db.delete_document.side_effect = Exception("table not found")
    db._open_table.return_value = None
    proc = DocumentProcessor(mock_chunker, mock_embedding, db)
    result = proc.process_document("doc1", "some text")
    assert len(result) == 2
    db.add_documents_from_df.assert_called_once()


def test_process_document_reraises_delete_when_table_exists(
    mock_chunker, mock_embedding
):
    """When delete_document raises and the table DOES exist, the error
    must propagate — not be silently swallowed."""
    db = MagicMock()
    db.delete_document.side_effect = RuntimeError("real db error")
    db._open_table.return_value = MagicMock()  # table exists
    proc = DocumentProcessor(mock_chunker, mock_embedding, db)
    with pytest.raises(RuntimeError, match="real db error"):
        proc.process_document("doc1", "some text")
