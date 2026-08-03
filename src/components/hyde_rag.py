import logging
from typing import List, Optional, Tuple

import dspy

from .bm25_manager import BM25Manager
from .embedding_manager import EmbeddingManager
from .lancedb_manager import LanceDBManager

logger = logging.getLogger(__name__)


class GenerateHypotheticalDoc(dspy.Signature):
    """Generates a detailed hypothetical answer to the question to support
    document retrieval."""

    question = dspy.InputField()
    hypothetical_document = dspy.OutputField(
        desc="A realistic-sounding passage that answers the question."
    )


class GenerateQueries(dspy.Signature):
    """Produces up to N reformulations of the question to broaden retrieval coverage.

    Each reformulation should rephrase the same intent using different terminology,
    so vector search can match documents phrased differently from the original query.
    """

    question = dspy.InputField()
    queries = dspy.OutputField(
        desc="A newline-separated list of reformulated questions (one per line)."
    )


class PredictTerms(dspy.Signature):
    """Predicts terms likely to appear verbatim in the answer document but
    NOT in the query.

    This is SIRA-style query-side term prediction: think of concrete identifiers,
    error codes, version numbers, product names, or technical terms the answer
    document would contain. Return a comma-separated list.
    """

    question = dspy.InputField()
    terms = dspy.OutputField(
        desc="Comma-separated list of predicted answer-document terms."
    )


class FinalRAG(dspy.Signature):
    """Answers the question based on the provided real contexts."""

    context = dspy.InputField(desc="Relevant real documents from the database.")
    question = dspy.InputField()
    answer = dspy.OutputField()


def _rrf_fuse(
    ranked_lists: List[List[Tuple[str, str, float]]], k_const: int = 60, limit: int = 5
) -> List[Tuple[str, str, float]]:
    """Reciprocal Rank Fusion of several ranked result lists.

    Each input list is a sequence of (chunk_id, text, score) best-first.
    Returns a fused list of (chunk_id, text, rrf_score) best-first.
    """
    scores: dict = {}
    text_by_id: dict = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, text, _score) in enumerate(ranked):
            rrf = 1.0 / (k_const + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf
            text_by_id.setdefault(chunk_id, text)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(cid, text_by_id[cid], s) for cid, s in fused[:limit]]


class HyDERAG(dspy.Module):
    def __init__(
        self,
        db_manager: LanceDBManager,
        embedding_manager: EmbeddingManager,
        bm25_manager: Optional[BM25Manager] = None,
        multi_query: bool = True,
        num_queries: int = 5,
        bm25_hybrid: bool = True,
    ):
        super().__init__()
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.bm25_manager = bm25_manager
        self.multi_query = multi_query
        self.num_queries = num_queries
        self.bm25_hybrid = bm25_hybrid and bm25_manager is not None
        # DSPy uses ChainOfThought to give the model time to "think"
        self.generate_hyde = dspy.ChainOfThought(GenerateHypotheticalDoc)
        self.generate_queries = dspy.ChainOfThought(GenerateQueries)
        self.predict_terms = dspy.ChainOfThought(PredictTerms)
        self.final_rag = dspy.ChainOfThought(FinalRAG)

    def forward(self, question: str, document_id: str = None, k: int = 3):
        # Step 1: generate hypothetical document (primary HyDE path)
        hyde_res = self.generate_hyde(question=question)

        # Step 2: retrieve contexts (multi-query + optional BM25 hybrid fusion)
        retrieved_contexts = self._retrieve_contexts(
            hyde_res.hypothetical_document,
            question=question,
            k=k,
            document_id=document_id,
        )

        # Step 3: generate the real answer based on the retrieved documents
        final_res = self.final_rag(context=retrieved_contexts, question=question)

        return dspy.Prediction(answer=final_res.answer, context=retrieved_contexts)

    def _retrieve_contexts(
        self,
        hypothetical_document: str,
        question: str = None,
        k: int = 3,
        document_id: str = None,
    ) -> str:
        """
        Retrieves relevant contexts from the database.

        Combines vector search over the HyDE document (and, when multi_query is
        enabled, over LLM-reformulated queries) with optional SIRA-style
        weighted BM25 retrieval, fused via Reciprocal Rank Fusion.

        :param hypothetical_document: The hypothetical document text
        :param question: Original user question (for multi-query and BM25)
        :param k: Number of contexts to retrieve
        :param document_id: Optional filter on a specific document ID
        :return: Concatenated contexts as a string
        :raises RuntimeError: If the embedding API fails
        """
        # --- Vector retrieval (HyDE) ---
        vector_texts = self._vector_search(
            hypothetical_document, k=k, document_id=document_id
        )

        # --- Multi-query vector retrieval ---
        if self.multi_query and question:
            queries = self._reformulate_queries(question)
            for q in queries:
                vector_texts.extend(
                    self._vector_search_text(q, k=k, document_id=document_id)
                )

        # --- BM25 hybrid retrieval (SIRA query compilation) ---
        bm25_texts: List[str] = []
        if self.bm25_hybrid and self.bm25_manager is not None and question:
            bm25_texts = self._bm25_search(question, k=k, document_id=document_id)

        # --- Fusion (RRF over vector + BM25) or simple dedup ---
        if self.bm25_hybrid and bm25_texts:
            fused = self._fuse(vector_texts, bm25_texts, k=k)
            contexts = fused
        else:
            contexts = vector_texts

        # Remove duplicates while preserving order
        unique_contexts = list(dict.fromkeys(contexts))
        if not unique_contexts:
            return "No relevant contexts found."
        return "\n\n".join(unique_contexts)

    def _vector_search(
        self, hypothetical_document: str, k: int, document_id: Optional[str]
    ) -> List[str]:
        """Vector search using the HyDE hypothetical document."""
        if not hypothetical_document:
            return []
        try:
            embeddings = self.embedding_manager.get_embeddings([hypothetical_document])
        except RuntimeError as e:
            logger.warning("HyDE embedding failed, skipping vector search: %s", e)
            return []
        query_vector = embeddings[0]
        return self._search_to_texts(query_vector, limit=k, document_id=document_id)

    def _vector_search_text(
        self, query_text: str, k: int, document_id: Optional[str]
    ) -> List[str]:
        """Vector search using arbitrary text (e.g. a reformulated query)."""
        if not query_text:
            return []
        try:
            embeddings = self.embedding_manager.get_embeddings([query_text])
        except RuntimeError as e:
            logger.warning("Multi-query embedding failed: %s", e)
            return []
        return self._search_to_texts(embeddings[0], limit=k, document_id=document_id)

    def _search_to_texts(
        self, query_vector: List[float], limit: int, document_id: Optional[str]
    ) -> List[str]:
        results = self.db_manager.search(
            query_vector, limit=limit, document_id=document_id
        )
        if not results.empty and "text" in results.columns:
            return results["text"].tolist()
        return []

    def _reformulate_queries(self, question: str) -> List[str]:
        """Ask the LLM for up to num_queries reformulations of the question."""
        try:
            res = self.generate_queries(question=question)
        except Exception as e:
            logger.warning("Multi-query reformulation failed: %s", e)
            return []
        raw = (res.queries or "").strip()
        if not raw:
            return []
        queries = [q.strip() for q in raw.splitlines() if q.strip()]
        return queries[: self.num_queries]

    def _bm25_search(
        self, question: str, k: int, document_id: Optional[str]
    ) -> List[str]:
        """SIRA BM25: predict terms, DF-filter, weighted BM25, return texts."""
        try:
            terms_res = self.predict_terms(question=question)
        except Exception as e:
            logger.warning("Term prediction failed: %s", e)
            return []
        raw_terms = (terms_res.terms or "").split(",")
        candidate_terms = [t.strip() for t in raw_terms if t.strip()]
        if not candidate_terms:
            return []
        filtered = self.bm25_manager.filter_expansion_terms(candidate_terms)
        results = self.bm25_manager.search(
            original_query=question,
            expansion_terms=filtered,
            limit=k,
            document_id=document_id,
        )
        return [text for (_cid, text, _score) in results]

    def _fuse(
        self, vector_texts: List[str], bm25_texts: List[str], k: int
    ) -> List[str]:
        """RRF-fuse vector and BM25 text lists into one ranked text list.

        The scores in the wrapped tuples are placeholders (``0.0``): RRF uses
        only the *rank* of each item, never its underlying score, so the actual
        values are ignored by ``_rrf_fuse``. ``chunk_id`` is set to the text
        itself, so deduplication happens by text (two identical chunks collapse
        into one, which is the intended behaviour for fusing two rankings of
        overlapping content).
        """
        vec_ranked = [(t, t, 0.0) for t in vector_texts]
        bm25_ranked = [(t, t, 0.0) for t in bm25_texts]
        fused = _rrf_fuse([vec_ranked, bm25_ranked], limit=k * 2)
        return [text for (_cid, text, _score) in fused]
