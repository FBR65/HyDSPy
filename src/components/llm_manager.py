import logging
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMManager:
    def __init__(self, base_url: str, api_key: str, model_id: str):
        """
        Initialises the LLM manager.

        :param base_url: Base URL of the LLM
        :param api_key: API key for the LLM
        :param model_id: ID of the LLM
        :raises RuntimeError: On any LLM API failure
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id

    def generate_completion(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generates a completion for a given prompt.

        Implemented via the chat-completions endpoint (``chat.completions``)
        rather than the legacy text-completions endpoint, because many
        OpenAI-compatible servers (e.g. Ollama) do not support the legacy
        ``/v1/completions`` route. The prompt is sent as a single user message.

        :param prompt: Input prompt
        :param max_tokens: Maximum number of tokens in the response
        :return: Generated completion
        :raises RuntimeError: If the LLM API fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as e:
            raise RuntimeError(
                f"LLM completion for model '{self.model_id}' failed: {e}"
            ) from e

    def generate_chat_completion(
        self, messages: List[dict], max_tokens: int = 500
    ) -> str:
        """
        Generates a chat completion for a list of messages.

        :param messages: List of messages (dialogue history)
        :param max_tokens: Maximum number of tokens in the response
        :return: Generated chat completion
        :raises RuntimeError: If the LLM API fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_id, messages=messages, max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as e:
            raise RuntimeError(
                f"LLM chat completion for model '{self.model_id}' failed: {e}"
            ) from e
