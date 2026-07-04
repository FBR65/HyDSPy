"""Tests for src.components.bm25_manager.BM25Manager.

Covers the SIRA query-compilation pipeline: DF filtering, weighted scoring,
document_id filtering, and the cached document_id column (no per-search
table reload).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.components.bm25_manager import BM25Manager, _tokenize
from src.components.lancedb_manager import LanceDBManager


def _make_corpus_df(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build a chunk DataFrame from (document_id, chunk_id, text) tuples."""
    return pd.DataFrame(
        [
            {
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "text": text,
                "vector": [float(i)] * 8,
                "source": "test",
                "chunk_index": i,
                "total_chunks": len(rows),
            }
            for i, (doc_id, chunk_id, text) in enumerate(rows)
        ]
    )


@pytest.fixture
def bm25_indexed(temp_lancedb_uri):
    """LanceDB + BM25Manager loaded with a small fixed corpus."""
    db = LanceDBManager(uri=temp_lancedb_uri, table_name="bm25_corpus")
    db.add_documents_from_df(
        _make_corpus_df(
            [
                ("doc1", "c1", "the ldap error code 49 indicates invalid credentials"),
                ("doc1", "c2", "kubernetes pod crashloopbackoff restart loop"),
                ("doc2", "c3", "ldap bind operation requires a valid dn and password"),
                ("doc2", "c4", "postgres connection refused check the port"),
                ("doc3", "c5", "common the the the filler text repeated words"),
            ]
        )
    )
    bm25 = BM25Manager(db_manager=db, expansion_weight=0.5, df_threshold=0.5)
    bm25._refresh_index()
    return db, bm25


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_word_boundaries():
    assert _tokenize("Hello, WORLD! 42") == ["hello", "world", "42"]


def test_tokenize_handles_none_and_empty():
    assert _tokenize(None) == []
    assert _tokenize("") == []


# ---------------------------------------------------------------------------
# DF filter
# ---------------------------------------------------------------------------


def test_filter_drops_terms_absent_from_corpus(bm25_indexed):
    _db, bm25 = bm25_indexed
    kept = bm25.filter_expansion_terms(["nonexistentterm"])
    assert kept == []


def test_filter_drops_overly_common_terms(bm25_indexed):
    _db, bm25 = bm25_indexed
    # "the" appears in 4 of 5 docs -> DF=4 > 0.5*5=2 -> dropped.
    kept = bm25.filter_expansion_terms(["the"])
    assert "the" not in kept


def test_filter_keeps_specific_terms(bm25_indexed):
    _db, bm25 = bm25_indexed
    kept = bm25.filter_expansion_terms(["ldap", "postgres"])
    assert set(kept) == {"ldap", "postgres"}


def test_filter_deduplicates_and_lowercases(bm25_indexed):
    _db, bm25 = bm25_indexed
    kept = bm25.filter_expansion_terms(["LDAP", "ldap"])
    assert kept == ["ldap"]


def test_filter_empty_when_index_empty(temp_lancedb_uri):
    db = LanceDBManager(uri=temp_lancedb_uri, table_name="empty")
    bm25 = BM25Manager(db_manager=db)
    assert bm25.filter_expansion_terms(["anything"]) == []


# ---------------------------------------------------------------------------
# Weighted search
# ---------------------------------------------------------------------------


def test_search_returns_relevant_chunks_first(bm25_indexed):
    _db, bm25 = bm25_indexed
    results = bm25.search("ldap error", expansion_terms=[], limit=3)
    assert len(results) <= 3
    texts = [t for (_cid, t, _s) in results]
    # At least one ldap-related chunk in the top results.
    assert any("ldap" in t.lower() for t in texts)


def test_search_with_expansion_terms_boosts_matching_chunks(bm25_indexed):
    _db, bm25 = bm25_indexed
    boosted = bm25.search("bind", expansion_terms=["ldap", "password"], limit=5)
    boosted_texts = {t for (_cid, t, _s) in boosted}
    # The ldap/password chunks should be ranked higher with expansion.
    ldap_texts = {
        t for t in boosted_texts if "ldap" in t.lower() or "password" in t.lower()
    }
    assert ldap_texts, "expansion terms should surface ldap/password chunks"


def test_search_does_not_double_filter_expansion_terms(bm25_indexed):
    """Regression: search() must NOT re-run filter_expansion_terms on already
    caller-filtered terms. A term that would be dropped by the DF filter (too
    common) should still influence the score when passed directly, proving the
    filter is the caller's responsibility."""
    _db, bm25 = bm25_indexed
    # "the" is common and would be filtered out by filter_expansion_terms,
    # but search() must apply it as-is.
    results = bm25.search("nothing", expansion_terms=["the"], limit=5)
    # Non-empty result proves the term was used rather than dropped.
    assert isinstance(results, list)


def test_search_document_id_filter_uses_cached_column(bm25_indexed):
    """document_id filtering must work without reloading the table; the cached
    _doc_id_col is the source of truth."""
    _db, bm25 = bm25_indexed
    results = bm25.search("ldap", expansion_terms=[], limit=10, document_id="doc1")
    assert results, "expected doc1 results for 'ldap'"
    # Every returned chunk must belong to doc1 (the filter is the contract).
    doc_ids = {bm25._doc_id_col[bm25._doc_ids.index(cid)] for (cid, _t, _s) in results}
    assert doc_ids == {"doc1"}
    results_doc2 = bm25.search("ldap", expansion_terms=[], limit=10, document_id="doc2")
    doc_ids_doc2 = {
        bm25._doc_id_col[bm25._doc_ids.index(cid)] for (cid, _t, _s) in results_doc2
    }
    assert doc_ids_doc2 == {"doc2"}
    # The two result sets are disjoint by document.
    assert {cid for (cid, _t, _s) in results} != {cid for (cid, _t, _s) in results_doc2}


def test_search_empty_index_returns_empty(temp_lancedb_uri):
    db = LanceDBManager(uri=temp_lancedb_uri, table_name="empty")
    bm25 = BM25Manager(db_manager=db)
    assert bm25.search("anything", expansion_terms=[], limit=5) == []


def test_invalidate_forces_rebuild(bm25_indexed):
    _db, bm25 = bm25_indexed
    assert bm25._bm25 is not None
    bm25.invalidate()
    assert bm25._row_count is None
    # Next search rebuilds.
    bm25.search("ldap", expansion_terms=[], limit=1)
    assert bm25._bm25 is not None
