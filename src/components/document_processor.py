from typing import Any, Dict, List

import pandas as pd

from .chunker import SemanticChunker
from .embedding_manager import EmbeddingManager
from .lancedb_manager import LanceDBManager


class DocumentProcessor:
    def __init__(
        self,
        chunker: SemanticChunker,
        embedding_manager: EmbeddingManager,
        db_manager: LanceDBManager,
    ):
        """
        Initialises the document processor.

        :param chunker: SemanticChunker for text segmentation
        :param embedding_manager: EmbeddingManager for vectorisation
        :param db_manager: LanceDBManager for database storage
        """
        self.chunker = chunker
        self.embedding_manager = embedding_manager
        self.db_manager = db_manager

    def process_document(self, document_id: str, text: str) -> List[Dict[str, Any]]:
        """
        Processes a document through chunking, vectorisation, and storage.

        Idempotent: a subsequent call with the same ``document_id`` first
        removes the existing chunks before inserting the new ones, so repeated
        calls never produce duplicates.

        :param document_id: Unique ID of the document
        :param text: Document text
        :return: List of processed chunks with metadata
        """
        # 1. Chunking
        chunks = self.chunker.chunk(text)
        if not chunks:
            return []

        # 2. Create embeddings
        embeddings = self.embedding_manager.get_embeddings(chunks)

        # 3. Prepare chunks with metadata
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = {
                "document_id": document_id,
                "chunk_id": f"{document_id}_chunk_{i}",
                "text": chunk,
                "vector": embedding,  # Vector column for LanceDB
                "source": "processed_document",
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            documents.append(doc)

        # 4. Store in database.
        # Remove existing chunks of the same document so a subsequent call
        # does not create duplicates (idempotency). We only swallow the
        # "table does not exist yet" case — genuine errors are re-raised.
        try:
            self.db_manager.delete_document(document_id)
        except Exception:
            if self.db_manager._open_table() is None:
                pass  # Table does not exist yet — expected on first run.
            else:
                raise  # Real error — do not silently swallow.
        self.db_manager.add_documents_from_df(pd.DataFrame(documents))

        return documents
