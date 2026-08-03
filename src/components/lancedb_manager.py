import logging
from typing import List, Optional

import pandas as pd

import lancedb

logger = logging.getLogger(__name__)


def _escape_sql_literal(value: str) -> str:
    """Escapes a string for safe embedding in a LanceDB SQL-style filter.

    LanceDB uses SQL-like ``WHERE`` clauses with single-quoted string literals.
    A ``'`` inside the value would break the filter or allow filter injection.
    Single quotes are doubled (``'`` -> ``''``), the standard SQL escape rule.
    """
    return str(value).replace("'", "''")


class LanceDBManager:
    def __init__(
        self,
        uri: str,
        table_name: str,
        index_type: str = "IVF_PQ",
        num_partitions: Optional[int] = None,
        num_sub_vectors: Optional[int] = None,
        index_train_threshold: int = 256,
    ):
        """
        Initialises the LanceDB manager.

        :param uri: URI to the LanceDB database
        :param table_name: Name of the table in the database
        :param index_type: Vector index type (e.g. ``IVF_PQ``, ``IVF_FLAT``,
            ``IVF_HNSW_SQ``, ``IVF_HNSW_PQ``). ``None`` disables index build;
            LanceDB then uses a sequential scan.
        :param num_partitions: Number of IVF partitions. ``None`` leaves the
            choice to LanceDB (default: sqrt(N)). A small value (e.g. 16) is
            recommended for small corpora.
        :param num_sub_vectors: Number of PQ sub-vectors. ``None`` leaves the
            choice to LanceDB. Only relevant for ``IVF_PQ`` / ``IVF_HNSW_PQ``.
        :param index_train_threshold: Minimum row count before a vector index
            is built. Below this threshold the default index (sequential scan)
            is used, as IVF training can be unstable on small datasets.
        """
        self.db = lancedb.connect(uri)
        self.table_name = table_name
        self.table = None
        self.index_type = index_type
        self.num_partitions = num_partitions
        self.num_sub_vectors = num_sub_vectors
        self.index_train_threshold = index_train_threshold
        self._index_built = False

    def _open_table(self):
        """Opens the table, returning None if it does not exist."""
        try:
            self.table = self.db.open_table(self.table_name)
            return self.table
        except Exception:
            self.table = None
            return None

    def create_table_from_df(self, df: pd.DataFrame):
        """
        Creates a new table in the database from a DataFrame.

        :param df: DataFrame with the data
        """
        self.table = self.db.create_table(self.table_name, data=df)
        self._maybe_build_index()
        return self.table

    def get_or_create_table_from_df(self, df: pd.DataFrame):
        """
        Retrieves an existing table or creates it from a DataFrame.

        :param df: DataFrame with the data
        :return: LanceDB table
        """
        if self._open_table() is None:
            self.table = self.db.create_table(self.table_name, data=df)
        self._maybe_build_index()
        return self.table

    def add_documents_from_df(self, df: pd.DataFrame):
        """
        Adds documents to the table from a DataFrame.

        Resets the ``_index_built`` flag so the vector index is rebuilt by the
        next ``_maybe_build_index()`` call once the row count crosses the
        threshold. This prevents a stale index on growing corpora.
        """
        if self._open_table() is None:
            self.table = self.db.create_table(self.table_name, data=df)
        else:
            self.table.add(df)
        # New data may invalidate an existing index; rebuild on next call.
        self._index_built = False
        self._maybe_build_index()

    def _maybe_build_index(self) -> None:
        """Builds the configured vector index once enough data is available.
        Idempotent: an already-built index is not recreated. For growing
        corpora a manual ``rebuild_index()`` is recommended."""
        if self.index_type is None or self._index_built:
            return
        table = self._open_table()
        if table is None:
            return
        try:
            n = table.count_rows()
        except Exception:
            return
        if n < self.index_train_threshold:
            return
        try:
            kwargs = {"index_type": self.index_type, "replace": True}
            if self.num_partitions is not None:
                kwargs["num_partitions"] = self.num_partitions
            if self.num_sub_vectors is not None:
                kwargs["num_sub_vectors"] = self.num_sub_vectors
            table.create_index(**kwargs)
            self._index_built = True
            logger.info("Vector index '%s' built (%d rows).", self.index_type, n)
        except Exception as e:
            logger.warning("Vector index build '%s' failed: %s", self.index_type, e)

    def rebuild_index(self) -> None:
        """Forces a rebuild of the vector index (e.g. after large insert
        operations)."""
        self._index_built = False
        self._maybe_build_index()

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Searches for similar vectors in the database.

        :param query_vector: Query vector
        :param limit: Maximum number of results
        :param document_id: Optional filter on the 'document_id' field
        :return: Search results as a DataFrame
        """
        table = self._open_table()
        if table is None:
            return pd.DataFrame()
        try:
            query = table.search(query_vector)
            if document_id is not None:
                query = query.where(
                    f"document_id = '{_escape_sql_literal(document_id)}'"
                )
            return query.limit(limit).to_pandas()
        except Exception as e:
            logger.warning("Search in table '%s' failed: %s", self.table_name, e)
            return pd.DataFrame()

    def delete_document(self, document_id: str) -> int:
        """
        Removes all chunks with the given document_id from the table.

        :param document_id: ID of the document to delete
        :return: Number of removed rows
        """
        table = self._open_table()
        if table is None:
            return 0
        try:
            count_before = table.count_rows()
            table.delete(f"document_id = '{_escape_sql_literal(document_id)}'")
            count_after = table.count_rows()
            return count_before - count_after
        except Exception as e:
            logger.error("Delete of document_id='%s' failed: %s", document_id, e)
            raise

    def get_document_chunks(self, document_id: str) -> pd.DataFrame:
        """
        Returns all chunks of a document, sorted by ``chunk_index``.

        Public alternative to direct ``_open_table()`` / ``table.to_pandas()``
        access so callers (e.g. the agentic ``open`` tool) do not need to
        touch LanceDB internals.

        :param document_id: ID of the document
        :return: DataFrame with the document's chunks (empty if absent or
            the table does not exist)
        """
        table = self._open_table()
        if table is None:
            return pd.DataFrame()
        try:
            df = table.to_pandas()
        except Exception as e:
            logger.warning("get_document_chunks failed: %s", e)
            return pd.DataFrame()
        if df.empty or "document_id" not in df.columns:
            return pd.DataFrame()
        if "chunk_index" in df.columns:
            df = df.sort_values("chunk_index")
        return df[df["document_id"].astype(str) == str(document_id)].reset_index(
            drop=True
        )

    def list_documents(self) -> List[dict]:
        """
        Returns an overview of all indexed documents.

        :return: List of dicts with keys: document_id, chunk_count,
            first_chunk_index, last_chunk_index
        """
        table = self._open_table()
        if table is None:
            return []
        try:
            df = table.to_pandas()
        except Exception as e:
            logger.warning("list_documents failed: %s", e)
            return []
        if df.empty or "document_id" not in df.columns:
            return []
        agg = (
            df.groupby("document_id")
            .agg(
                chunk_count=("chunk_id", "count"),
                first_chunk_index=("chunk_index", "min"),
                last_chunk_index=("chunk_index", "max"),
            )
            .reset_index()
        )
        return agg.to_dict(orient="records")
