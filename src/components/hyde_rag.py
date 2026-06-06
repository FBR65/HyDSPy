import logging
import dspy
from typing import List
from .lancedb_manager import LanceDBManager
from .embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)


class GenerateHypotheticalDoc(dspy.Signature):
    """Generates a detailed hypothetical answer to the question to support document retrieval."""
    question = dspy.InputField()
    hypothetical_document = dspy.OutputField(desc="A realistic-sounding passage that answers the question.")


class FinalRAG(dspy.Signature):
    """Answers the question based on the provided real contexts."""
    context = dspy.InputField(desc="Relevant real documents from the database.")
    question = dspy.InputField()
    answer = dspy.OutputField()


class HyDERAG(dspy.Module):
    def __init__(self, db_manager: LanceDBManager, embedding_manager: EmbeddingManager):
        super().__init__()
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        # DSPy uses ChainOfThought to give the model time to "think"
        self.generate_hyde = dspy.ChainOfThought(GenerateHypotheticalDoc)
        self.final_rag = dspy.ChainOfThought(FinalRAG)

    def forward(self, question: str, document_id: str = None, k: int = 3):
        # Step 1: generate hypothetical document
        hyde_res = self.generate_hyde(question=question)

        # Step 2: retrieve using the hypothetical text
        retrieved_contexts = self._retrieve_contexts(
            hyde_res.hypothetical_document, k=k, document_id=document_id
        )

        # Step 3: generate the real answer based on the retrieved documents
        final_res = self.final_rag(context=retrieved_contexts, question=question)

        return dspy.Prediction(answer=final_res.answer, context=retrieved_contexts)

    def _retrieve_contexts(
        self,
        hypothetical_document: str,
        k: int = 3,
        document_id: str = None,
    ) -> str:
        """
        Retrieves relevant contexts from the database.

        :param hypothetical_document: The hypothetical document text
        :param k: Number of contexts to retrieve
        :param document_id: Optional filter on a specific document ID
        :return: Concatenated contexts as a string
        :raises RuntimeError: If the embedding API fails
        """
        # Create embedding for the hypothetical document
        embeddings = self.embedding_manager.get_embeddings([hypothetical_document])
        query_vector = embeddings[0]

        # Search the database for similar documents
        results = self.db_manager.search(
            query_vector, limit=k, document_id=document_id
        )
        if not results.empty and 'text' in results.columns:
            contexts = results['text'].tolist()
            # Remove duplicates
            unique_contexts = list(dict.fromkeys(contexts))
            return "\n\n".join(unique_contexts)
        return "No relevant contexts found."
