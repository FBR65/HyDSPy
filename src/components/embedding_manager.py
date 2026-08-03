import logging
from typing import List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# Default batch size for embedding requests. Many OpenAI-compatible endpoints
# limit the number of input items per request (e.g. 2048 for OpenAI); 64 is a
# safe default that keeps individual requests small enough for local servers
# like Ollama while avoiding a single huge call for large corpora.
_DEFAULT_BATCH_SIZE = 64


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

        Requests are sent in batches (default 64 items) to stay within the
        input limits of OpenAI-compatible endpoints and to avoid one giant
        request for large corpora. Results are concatenated in order.

        :param texts: List of texts to embed
        :return: List of embeddings
        :raises RuntimeError: If the embedding API fails
        """
        if not texts:
            return []
        vectors: List[List[float]] = []
        for start in range(0, len(texts), _DEFAULT_BATCH_SIZE):
            batch = texts[start : start + _DEFAULT_BATCH_SIZE]
            try:
                response = self.client.embeddings.create(
                    input=batch, model=self.model_id
                )
            except Exception as e:
                raise RuntimeError(
                    f"Embedding creation for model '{self.model_id}' "
                    f"failed at batch {start}-{start + len(batch)}: {e}"
                ) from e
            for item in response.data:
                vectors.append(item.embedding)
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors
