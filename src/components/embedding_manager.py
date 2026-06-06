import logging
from openai import OpenAI
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingManager:
    def __init__(self, base_url: str, api_key: str, model_id: str):
        """
        Initialises the embedding manager.

        :param base_url: Base URL of the embedding model
        :param api_key: API key for the embedding model
        :param model_id: ID of the embedding model
        :raises RuntimeError: On any embedding API failure
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError(
                "Embedding dimension has not been determined yet. "
                "Call get_embeddings() first."
            )
        return self._dimension

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Creates embeddings for a list of texts.

        :param texts: List of texts to embed
        :return: List of embeddings
        :raises RuntimeError: If the embedding API fails
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_id
            )
        except Exception as e:
            raise RuntimeError(
                f"Embedding creation for model '{self.model_id}' failed: {e}"
            ) from e

        vectors = [item.embedding for item in response.data]
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors
