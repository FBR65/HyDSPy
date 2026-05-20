from openai import OpenAI
from typing import Dict, Any, List

class LLMManager:
    def __init__(self, base_url: str, api_key: str, model_id: str):
        """
        Initialisiert den LLM-Manager.
        
        :param base_url: Basis-URL des LLM
        :param api_key: API-Schlüssel für das LLM
        :param model_id: ID des LLM
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
    
    def generate_completion(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generiert eine Vervollständigung für einen gegebenen Prompt.
        
        :param prompt: Eingabeaufforderung
        :param max_tokens: Maximale Anzahl von Tokens in der Antwort
        :return: Generierte Vervollständigung
        """
        try:
            response = self.client.completions.create(
                model=self.model_id,
                prompt=prompt,
                max_tokens=max_tokens
            )
            return response.choices[0].text.strip()
        except Exception as e:
            # Falls ein Fehler auftritt, geben wir einen leeren String zurück
            return ""
    
    def generate_chat_completion(self, messages: list, max_tokens: int = 500) -> str:
        """
        Generiert eine Chat-Vervollständigung für eine Liste von Nachrichten.
        
        :param messages: Liste von Nachrichten (Dialogverlauf)
        :param max_tokens: Maximale Anzahl von Tokens in der Antwort
        :return: Generierte Chat-Vervollständigung
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Falls ein Fehler auftritt, geben wir einen leeren String zurück
            return ""