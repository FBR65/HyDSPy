import logging
import lancedb
import pandas as pd
from typing import List, Optional

logger = logging.getLogger(__name__)


class LanceDBManager:
    def __init__(self, uri: str, table_name: str):
        """
        Initialisiert den LanceDB-Manager.

        :param uri: URI zur LanceDB-Datenbank
        :param table_name: Name der Tabelle in der Datenbank
        """
        self.db = lancedb.connect(uri)
        self.table_name = table_name
        self.table = None

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
        return self.table

    def get_or_create_table_from_df(self, df: pd.DataFrame):
        """
        Ruft eine vorhandene Tabelle ab oder erstellt sie aus einem DataFrame.

        :param df: DataFrame mit den Daten
        :return: LanceDB-Tabelle
        """
        if self._open_table() is None:
            self.table = self.db.create_table(self.table_name, data=df)
        return self.table

    def add_documents_from_df(self, df: pd.DataFrame):
        """
        Fügt Dokumente zur Tabelle aus einem DataFrame hinzu.

        :param df: DataFrame mit den Dokumenten
        """
        if self._open_table() is None:
            self.table = self.db.create_table(self.table_name, data=df)
        else:
            self.table.add(df)

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
                query = query.where(f"document_id = '{document_id}'")
            return query.limit(limit).to_pandas()
        except Exception as e:
            logger.warning("Suche in Tabelle '%s' fehlgeschlagen: %s", self.table_name, e)
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
            table.delete(f"document_id = '{document_id}'")
            count_after = table.count_rows()
            return count_before - count_after
        except Exception as e:
            logger.error(
                "Löschen von document_id='%s' fehlgeschlagen: %s", document_id, e
            )
            raise

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
