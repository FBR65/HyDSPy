import pyarrow as pa
import pandas as pd
from typing import List, Dict, Any
from .chunker import SemanticChunker
from .embedding_manager import EmbeddingManager
from .lancedb_manager import LanceDBManager

class DocumentProcessor:
    def __init__(self, chunker: SemanticChunker, embedding_manager: EmbeddingManager, db_manager: LanceDBManager):
        """
        Initialisiert den DocumentProcessor.
        
        :param chunker: SemanticChunker für die Textsegmentierung
        :param embedding_manager: EmbeddingManager für Vektorisierung
        :param db_manager: LanceDBManager für Datenbankspeicherung
        """
        self.chunker = chunker
        self.embedding_manager = embedding_manager
        self.db_manager = db_manager
    
    def process_document(self, document_id: str, text: str) -> List[Dict[str, Any]]:
        """
        Verarbeitet ein Dokument durch Chunking, Vektorisierung und Speicherung.
        
        :param document_id: Eindeutige ID des Dokuments
        :param text: Dokumenttext
        :return: Liste der verarbeiteten Chunks mit Metadaten
        """
        # 1. Chunking
        chunks = self.chunker.chunk(text)
        
        # 2. Embedding erstellen
        embeddings = self.embedding_manager.get_embeddings(chunks)
        
        # 3. Chunks mit Metadaten vorbereiten
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = {
                "document_id": document_id,
                "chunk_id": f"{document_id}_chunk_{i}",
                "text": chunk,
                "vector": embedding,  # Vektorspalte für LanceDB
                "source": "processed_document",
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            documents.append(doc)
        
        # 4. In Datenbank speichern
        if documents:
            # Erstelle DataFrame
            df = pd.DataFrame(documents)
            
            # Füge Dokumente hinzu
            self.db_manager.add_documents_from_df(df)
        
        return documents