import lancedb
import pandas as pd
import pyarrow as pa
from typing import List, Dict, Any

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
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            self.table = self.db.create_table(self.table_name, data=df)
        return self.table
    
    def add_documents_from_df(self, df: pd.DataFrame):
        """
        Fügt Dokumente zur Tabelle aus einem DataFrame hinzu.
        
        :param df: DataFrame mit den Dokumenten
        """
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            self.table = self.db.create_table(self.table_name, data=df)
        
        # Füge die Daten hinzu
        self.table.add(df)
    
    def search(self, query_vector: List[float], limit: int = 5):
        """
        Sucht nach ähnlichen Vektoren in der Datenbank.
        
        :param query_vector: Abfragevektor
        :param limit: Maximale Anzahl von Ergebnissen
        :return: Suchergebnisse
        """
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            # Wenn keine Tabelle existiert, geben wir ein leeres Ergebnis zurück
            return pd.DataFrame()
        
        try:
            return self.table.search(query_vector).limit(limit).to_pandas()
        except Exception as e:
            print(f"Fehler bei der Suche: {e}")
            # Rückgabe eines leeren DataFrames als Fallback
            return pd.DataFrame()