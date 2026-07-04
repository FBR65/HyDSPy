import logging
import lancedb
import pandas as pd
from typing import List, Optional

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
        Initialisiert den LanceDB-Manager.

        :param uri: URI zur LanceDB-Datenbank
        :param table_name: Name der Tabelle in der Datenbank
        :param index_type: Vektor-Index-Typ (z.B. ``IVF_PQ``, ``IVF_FLAT``,
            ``IVF_HNSW_SQ``, ``IVF_HNSW_PQ``). ``None`` deaktiviert den
            Index-Build; LanceDB nutzt dann einen sequentiellen Scan.
        :param num_partitions: Anzahl der IVF-Partitionen. ``None`` überlässt
            die Wahl LanceDB (Default: sqrt(N)). Fuer kleine Korpora ist ein
            kleiner Wert (z.B. 16) sinnvoller.
        :param num_sub_vectors: Anzahl der PQ-Sub-Vektoren. ``None`` uberlässt
            die Wahl LanceDB. Nur relevant fuer ``IVF_PQ``/``IVF_HNSW_PQ``.
        :param index_train_threshold: Minimale Zeilenzahl, ab der ein
            Vektor-Index aufgebaut wird. Unter diesem Schwellwert bleibt es
            beim Default-Index (Sequenz-Scan), da IVF-Training sonst
            instabil wird.
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
        """Öffnet die Tabelle, gibt None zurück, falls sie nicht existiert."""
        try:
            self.table = self.db.open_table(self.table_name)
            return self.table
        except Exception:
            self.table = None
            return None

    def create_table_from_df(self, df: pd.DataFrame):
        """
        Erstellt eine neue Tabelle in der Datenbank aus einem DataFrame.

        :param df: DataFrame mit den Daten
        """
        self.table = self.db.create_table(self.table_name, data=df)
        self._maybe_build_index()
        return self.table

    def get_or_create_table_from_df(self, df: pd.DataFrame):
        """
        Ruft eine vorhandene Tabelle ab oder erstellt sie aus einem DataFrame.

        :param df: DataFrame mit den Daten
        :return: LanceDB-Tabelle
        """
        if self._open_table() is None:
            self.table = self.db.create_table(self.table_name, data=df)
        self._maybe_build_index()
        return self.table

    def add_documents_from_df(self, df: pd.DataFrame):
        """
        Fügt Dokumente zur Tabelle aus einem DataFrame hinzu.

        Setzt das ``_index_built``-Flag zurück, damit der Vektor-Index beim
        nächsten ``_maybe_build_index()``-Aufruf (falls die Zeilenzahl den
        Schwellwert erreicht) neu aufgebaut wird. Bei wachsenden Korpora
        verhindert das einen veralteten Index.
        """
        if self._open_table() is None:
            self.table = self.db.create_table(self.table_name, data=df)
        else:
            self.table.add(df)
        # Neue Daten -> ein vorhandener Index ist potentiell veraltet; beim
        # naechsten _maybe_build_index() wird er bei Bedarf neu aufgebaut.
        self._index_built = False
        self._maybe_build_index()

    def _maybe_build_index(self) -> None:
        """Baut den konfigurierten Vektor-Index auf, sobald genug Daten
        vorhanden sind. Idempotent: ein bereits gebauter Index wird nicht
        erneut erzeugt. Bei wachsenden Korpora ist ein manuelles
        ``rebuild_index()`` ratsam."""
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
            logger.info("Vektor-Index '%s' aufgebaut (%d Zeilen).", self.index_type, n)
        except Exception as e:
            logger.warning(
                "Aufbau des Vektor-Index '%s' fehlgeschlagen: %s",
                self.index_type,
                e,
            )

    def rebuild_index(self) -> None:
        """Erzwingt einen Neuaufbau des Vektor-Index (z.B. nach großen
        Einfüge-Operationen)."""
        self._index_built = False
        self._maybe_build_index()

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        document_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Sucht nach ähnlichen Vektoren in der Datenbank.

        :param query_vector: Abfragevektor
        :param limit: Maximale Anzahl von Ergebnissen
        :param document_id: Optionaler Filter auf das Feld 'document_id'
        :return: Suchergebnisse als DataFrame
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
            logger.warning(
                "Suche in Tabelle '%s' fehlgeschlagen: %s", self.table_name, e
            )
            return pd.DataFrame()

    def delete_document(self, document_id: str) -> int:
        """
        Entfernt alle Chunks mit dem angegebenen document_id aus der Tabelle.

        :param document_id: ID des zu löschenden Dokuments
        :return: Anzahl der entfernten Zeilen
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
            logger.error(
                "Löschen von document_id='%s' fehlgeschlagen: %s", document_id, e
            )
            raise

    def get_document_chunks(self, document_id: str) -> pd.DataFrame:
        """
        Liefert alle Chunks eines Dokuments, sortiert nach ``chunk_index``.

        Öffentliche Alternative zum direkten Zugriff auf ``_open_table()`` /
        ``table.to_pandas()``, damit Aufrufer (z.B. die agentic ``open``-Tool)
        nicht auf LanceDB-Internals zugreifen müssen.

        :param document_id: ID des Dokuments
        :return: DataFrame mit den Chunks des Dokuments (leer, falls nicht
            vorhanden oder die Tabelle nicht existiert)
        """
        table = self._open_table()
        if table is None:
            return pd.DataFrame()
        try:
            df = table.to_pandas()
        except Exception as e:
            logger.warning("get_document_chunks fehlgeschlagen: %s", e)
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
        Liefert eine Übersicht aller indizierten Dokumente.

        :return: Liste mit {document_id, chunk_count, first_chunk_index, last_chunk_index}
        """
        table = self._open_table()
        if table is None:
            return []
        try:
            df = table.to_pandas()
        except Exception as e:
            logger.warning("list_documents fehlgeschlagen: %s", e)
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
