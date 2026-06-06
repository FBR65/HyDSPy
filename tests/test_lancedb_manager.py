"""Tests fuer src.components.lancedb_manager.LanceDBManager.

Verwendet eine tempdir-Instanz - keine externen Dienste.
"""
import pandas as pd
import pytest

from src.components.lancedb_manager import LanceDBManager


def _make_docs(document_id: str, n_chunks: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "document_id": document_id,
                "chunk_id": f"{document_id}_chunk_{i}",
                "text": f"Inhalt von {document_id} chunk {i}",
                "vector": [0.0] * 8,
                "source": "test",
                "chunk_index": i,
                "total_chunks": n_chunks,
            }
            for i in range(n_chunks)
        ]
    )


def test_table_creation_and_add(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="t")
    mgr.add_documents_from_df(_make_docs("doc1", 3))
    assert mgr._open_table() is not None
    rows = mgr._open_table().to_pandas()
    assert len(rows) == 3
    assert set(rows["document_id"].unique()) == {"doc1"}


def test_add_appends_to_existing_table(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="t")
    mgr.add_documents_from_df(_make_docs("doc1", 2))
    mgr.add_documents_from_df(_make_docs("doc2", 3))
    rows = mgr._open_table().to_pandas()
    assert len(rows) == 5
    assert set(rows["document_id"].unique()) == {"doc1", "doc2"}


def test_search_returns_dataframe(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="t")
    mgr.add_documents_from_df(_make_docs("doc1", 2))
    results = mgr.search([0.0] * 8, limit=10)
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 2
    assert "text" in results.columns


def test_search_with_document_id_filter(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="t")
    mgr.add_documents_from_df(_make_docs("doc1", 2))
    mgr.add_documents_from_df(_make_docs("doc2", 3))
    results = mgr.search([0.0] * 8, limit=10, document_id="doc1")
    assert len(results) == 2
    assert set(results["document_id"].unique()) == {"doc1"}


def test_search_on_missing_table_returns_empty(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="nonexistent")
    results = mgr.search([0.0] * 8, limit=5)
    assert isinstance(results, pd.DataFrame)
    assert results.empty


def test_delete_document_removes_only_matching_chunks(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="t")
    mgr.add_documents_from_df(_make_docs("doc1", 2))
    mgr.add_documents_from_df(_make_docs("doc2", 3))
    removed = mgr.delete_document("doc1")
    assert removed == 2
    rows = mgr._open_table().to_pandas()
    assert set(rows["document_id"].unique()) == {"doc2"}


def test_delete_nonexistent_document_returns_zero(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="t")
    mgr.add_documents_from_df(_make_docs("doc1", 2))
    assert mgr.delete_document("does-not-exist") == 0


def test_delete_on_missing_table_returns_zero(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="nonexistent")
    assert mgr.delete_document("doc1") == 0


def test_list_documents_aggregates_chunks(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="t")
    mgr.add_documents_from_df(_make_docs("doc1", 2))
    mgr.add_documents_from_df(_make_docs("doc2", 4))
    docs = mgr.list_documents()
    assert {d["document_id"] for d in docs} == {"doc1", "doc2"}
    by_id = {d["document_id"]: d for d in docs}
    assert by_id["doc1"]["chunk_count"] == 2
    assert by_id["doc1"]["first_chunk_index"] == 0
    assert by_id["doc1"]["last_chunk_index"] == 1
    assert by_id["doc2"]["chunk_count"] == 4
    assert by_id["doc2"]["first_chunk_index"] == 0
    assert by_id["doc2"]["last_chunk_index"] == 3


def test_list_documents_empty_when_no_table(temp_lancedb_uri):
    mgr = LanceDBManager(uri=temp_lancedb_uri, table_name="nonexistent")
    assert mgr.list_documents() == []
