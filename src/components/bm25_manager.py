"""SIRA-style BM25 retrieval with LLM query expansion and a DF filter.

Implements the query-compilation idea from
"Before Another RAG Hop, Try Compiling the Query for BM25" (Akcam, 2026):

1. LLM predicts terms that are likely to appear verbatim in the answer
   document but not in the query (query-side term prediction).
2. A document-frequency (DF) filter drops expansion terms that are absent
   from the corpus or too common (DF > df_threshold * N).
3. A weighted Okapi-BM25 score combines the original and expanded query:
       score(q) = BM25(q_orig) + expansion_weight * BM25(q_exp)
4. Results are ranked by the combined score.

The module is intentionally dependency-light: it uses ``rank_bm25`` for
Okapi-BM25 and builds an in-memory token index from the stored chunk texts.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from typing import Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    """Lower-case word tokenizer used by both index and query."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Manager:
    """BM25 index over chunk texts with SIRA-style query compilation.

    The index is built from the LanceDB chunk texts and persisted to disk as a
    pickle file so it does not need to be rebuilt on every startup. The cache
    is invalidated automatically when the LanceDB row count changes, and can
    be forced via ``invalidate()``.
    """

    def __init__(
        self,
        db_manager,
        expansion_weight: float = 0.5,
        df_threshold: float = 0.1,
        cache_path: Optional[str] = None,
    ) -> None:
        """
        :param db_manager: LanceDBManager used to read chunk texts
        :param expansion_weight: weight ``w`` in ``BM25(q_orig) + w*BM25(q_exp)``
        :param df_threshold: max fraction of docs a term may occur in to survive
            the DF filter (``0 < DF <= df_threshold * N``)
        :param cache_path: optional path to a pickle file used to persist the
            index. ``None`` disables persistence (in-memory only).
        """
        self.db_manager = db_manager
        self.expansion_weight = expansion_weight
        self.df_threshold = df_threshold
        self.cache_path = cache_path
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[str] = []
        self._doc_ids: List[str] = []
        self._doc_id_col: List[str] = []
        self._df: Dict[str, int] = {}
        self._n_docs: int = 0
        self._row_count: Optional[int] = None

    def invalidate(self) -> None:
        """Mark the index as stale so it is rebuilt (and re-cached) on next use."""
        self._bm25 = None
        self._row_count = None
        self._doc_id_col = []

    def _refresh_index(self) -> None:
        """Rebuild the BM25 index if the LanceDB table changed. Loads from the
        on-disk cache when valid, otherwise rebuilds and writes the cache."""
        table = self.db_manager._open_table()
        if table is None:
            self._bm25 = None
            self._corpus = []
            self._doc_ids = []
            self._doc_id_col = []
            self._df = {}
            self._n_docs = 0
            self._row_count = None
            return
        try:
            count = table.count_rows()
        except Exception:
            count = None
        if self._bm25 is not None and count == self._row_count:
            return
        # Try loading from cache when the row count matches.
        if self._try_load_cache(count):
            return
        df = table.to_pandas()
        if df.empty or "text" not in df.columns:
            self._bm25 = None
            self._corpus = []
            self._doc_ids = []
            self._doc_id_col = []
            self._df = {}
            self._n_docs = 0
            self._row_count = count
            return
        self._corpus = [str(t) for t in df["text"].tolist()]
        self._doc_ids = (
            df["chunk_id"].astype(str).tolist()
            if "chunk_id" in df.columns
            else [str(i) for i in range(len(self._corpus))]
        )
        # Cache the document_id column so search() does not need to reload the
        # whole table for every document_id-filtered query.
        self._doc_id_col = (
            df["document_id"].astype(str).tolist()
            if "document_id" in df.columns
            else [""] * len(self._corpus)
        )
        tokenized = [_tokenize(c) for c in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._n_docs = len(self._corpus)
        self._df = {}
        for toks in tokenized:
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1
        self._row_count = count
        self._save_cache()
        logger.info("BM25 index rebuilt: %d documents", self._n_docs)

    def _try_load_cache(self, row_count: Optional[int]) -> bool:
        """Attempt to load the index from ``cache_path``. Returns True on hit."""
        if not self.cache_path or not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning("BM25 cache load failed (%s): %s", self.cache_path, e)
            return False
        if not isinstance(data, dict) or data.get("row_count") != row_count:
            return False
        self._bm25 = data["bm25"]
        self._corpus = data["corpus"]
        self._doc_ids = data["doc_ids"]
        self._doc_id_col = data.get("doc_id_col", [""] * len(self._corpus))
        self._df = data["df"]
        self._n_docs = data["n_docs"]
        self._row_count = row_count
        logger.info("BM25 index loaded from cache: %d documents", self._n_docs)
        return True

    def _save_cache(self) -> None:
        """Persist the current index to ``cache_path`` if configured."""
        if not self.cache_path or self._bm25 is None:
            return
        try:
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            payload = {
                "bm25": self._bm25,
                "corpus": self._corpus,
                "doc_ids": self._doc_ids,
                "doc_id_col": self._doc_id_col,
                "df": self._df,
                "n_docs": self._n_docs,
                "row_count": self._row_count,
            }
            with open(self.cache_path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            logger.warning("BM25 cache save failed (%s): %s", self.cache_path, e)

    def filter_expansion_terms(self, terms: Sequence[str]) -> List[str]:
        """Apply the SIRA DF filter: keep terms with ``0 < DF <= df_threshold*N``.

        Unknown terms (DF == 0) and overly common terms are dropped. Terms are
        lower-cased and de-duplicated before filtering.
        """
        self._refresh_index()
        if self._n_docs == 0:
            return []
        max_df = max(1, int(self.df_threshold * self._n_docs))
        seen: set = set()
        kept: List[str] = []
        for raw in terms:
            term = (raw or "").strip().lower()
            if not term or term in seen:
                continue
            seen.add(term)
            df = self._df.get(term, 0)
            if 0 < df <= max_df:
                kept.append(term)
        return kept

    def search(
        self,
        original_query: str,
        expansion_terms: Sequence[str],
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        """Run weighted BM25 and return ranked results.

        :param original_query: the raw user question
        :param expansion_terms: expansion terms to add to the query. They are
            applied as-is (the DF filter is the caller's responsibility; this
            keeps the single-source-of-truth for filtering in
            ``filter_expansion_terms``).
        :param limit: number of results to return
        :param document_id: optional document_id filter
        :return: list of (chunk_id, text, score) tuples, best first
        """
        self._refresh_index()
        if self._bm25 is None or self._n_docs == 0:
            return []
        q_orig = _tokenize(original_query)
        scores_orig = self._bm25.get_scores(q_orig) if q_orig else [0.0] * self._n_docs
        # NOTE: expansion_terms are NOT re-filtered here. Callers that want the
        # SIRA DF filter must call ``filter_expansion_terms`` first (as
        # ``HyDERAG._bm25_search`` does). This avoids a redundant double filter.
        expansion_tokens = [t.lower() for t in expansion_terms if t]
        if expansion_tokens:
            scores_exp = self._bm25.get_scores(expansion_tokens)
            scores = [
                o + self.expansion_weight * e for o, e in zip(scores_orig, scores_exp)
            ]
        else:
            scores = list(scores_orig)

        order = sorted(range(self._n_docs), key=lambda i: scores[i], reverse=True)
        results: List[Tuple[str, str, float]] = []
        # Use the cached document_id column instead of reloading the table on
        # every filtered search.
        doc_ids_col = self._doc_id_col
        for idx in order:
            if document_id is not None and doc_ids_col[idx] != document_id:
                continue
            if scores[idx] <= 0 and not q_orig:
                continue
            results.append((self._doc_ids[idx], self._corpus[idx], float(scores[idx])))
            if len(results) >= limit:
                break
        return results
