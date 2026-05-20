from openai import OpenAI
from typing import List

class EmbeddingManager:
    def __init__(self, base_url: str, api_key: str, model_id: str):
        """
        Initialisiert den Embedding-Manager.
        
        :param base_url: Basis-URL des Embedding-Modells
        :param api_key: API-Schlüssel für das Embedding-Modell
        :param model_id: ID des Embedding-Modells
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Erstellt Embeddings für eine Liste von Texten.
        
        :param texts: Liste von Texten zum Einbetten
        :return: Liste von Embeddings
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_id
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            # Fallback für Demo-Umgebung ohne echte API-Schlüssel
            print(f"Fehler bei der Embedding-Erstellung: {e}")
            # Rückgabe von Mock-Embeddings
            return [[0.1] * 1536 for _ in texts]  # Standardgröße für text-embedding-3-small