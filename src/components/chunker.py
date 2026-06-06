import json
import logging
import re
from typing import List, Optional

from .llm_manager import LLMManager

logger = logging.getLogger(__name__)


def _split_by_size(text: str, chunk_size: int) -> List[str]:
    """
    Deterministically splits text by paragraphs, then by sentences, then by words,
    then by characters, so every returned piece is <= chunk_size.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        if len(paragraph) <= chunk_size:
            pieces.append(paragraph)
            continue
        pieces.extend(_split_long(paragraph, chunk_size))

    return [p for p in pieces if p]


def _split_long(text: str, chunk_size: int) -> List[str]:
    """Splits a single text into pieces <= chunk_size.
    Sentence boundaries first, then word boundaries, then character boundaries."""
    if len(text) <= chunk_size:
        return [text]
    pieces: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not sentence:
            continue
        if len(sentence) <= chunk_size:
            pieces.append(sentence)
        else:
            pieces.extend(_split_words(sentence, chunk_size))
    return [p for p in pieces if p]


def _split_words(text: str, chunk_size: int) -> List[str]:
    """Splits text at word boundaries. Words longer than chunk_size are split
    character-by-character."""
    chunks: List[str] = []
    current = ""
    for word in text.split():
        if len(word) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(word), chunk_size):
                chunks.append(word[i : i + chunk_size])
            continue
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) > chunk_size and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class SemanticChunker:
    """
    Chunker that delegates topic-based segmentation to an LLM. The LLM
    decides both the boundaries and the size of each chunk based on semantic
    coherence. The resulting chunks are guaranteed to be <= chunk_size; if
    the LLM produces a piece that exceeds the cap, it is split further at
    sentence/word boundaries.

    Without an llm_manager, text is split deterministically (paragraphs,
    sentences, words).
    """

    _PROMPT_TEMPLATE = (
        "Split the following text into topically coherent sections. "
        "Each section should be self-contained and around {chunk_size} "
        "characters long. Group related content together; do not split in the "
        "middle of a thought. Return ONLY a JSON object of the form "
        '{{"chunks": ["...", "..."]}}. No comments, no markdown.\n\n'
        "TEXT:\n{text}"
    )

    def __init__(self, llm_manager: Optional[LLMManager] = None, chunk_size: int = 1000):
        """
        :param llm_manager: LLMManager for topic-based segmentation. When None,
            text is split deterministically (paragraphs, sentences, words).
        :param chunk_size: Target size (characters) the LLM is asked to aim
            for, and a hard upper bound for the final chunks. The LLM decides
            topic boundaries; this is a safety cap, not a target.
        """
        self.llm_manager = llm_manager
        self.chunk_size = chunk_size
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")

    def chunk(self, text: str, chunk_size: Optional[int] = None) -> List[str]:
        """
        Splits text into chunks.

        :param text: Input text
        :param chunk_size: Overrides the default ``chunk_size`` for this call
        :return: List of text chunks, each <= chunk_size characters
        :raises ValueError: If the LLM response is not valid JSON of the expected shape
        :raises RuntimeError: If the underlying LLM API call fails
        """
        if not text or not text.strip():
            return []

        size = chunk_size if chunk_size is not None else self.chunk_size
        if size <= 0:
            raise ValueError("chunk_size must be > 0.")

        if self.llm_manager is None:
            raw_chunks = _split_by_size(text, size)
        else:
            raw_chunks = self._llm_split(text, size)
        return self._enforce_size(raw_chunks, size)

    def _llm_split(self, text: str, chunk_size: int) -> List[str]:
        """Calls the LLM and parses its JSON response. Raises on bad output."""
        raw = self.llm_manager.generate_completion(
            prompt=self._PROMPT_TEMPLATE.format(text=text, chunk_size=chunk_size),
            max_tokens=min(2000, max(256, len(text) // 2)),
        )
        if not raw:
            raise ValueError("LLM returned an empty response.")
        return self._parse_chunks(raw)

    @staticmethod
    def _parse_chunks(raw: str) -> List[str]:
        """Parses the LLM response as JSON. Raises ValueError on bad shape."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise ValueError(f"LLM response is not valid JSON: {raw[:200]}") from e
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as e2:
                raise ValueError(f"LLM response is not valid JSON: {raw[:200]}") from e2
        chunks = data.get("chunks") if isinstance(data, dict) else None
        if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
            raise ValueError(
                f"LLM response missing a 'chunks' list of strings: {raw[:200]}"
            )
        return [c for c in (c.strip() for c in chunks) if c]

    @staticmethod
    def _enforce_size(chunks: List[str], chunk_size: int) -> List[str]:
        """Guarantees no chunk exceeds chunk_size; empty chunks are dropped."""
        result: List[str] = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                if chunk.strip():
                    result.append(chunk)
                continue
            for piece in _split_by_size(chunk, chunk_size):
                if piece.strip():
                    result.append(piece)
        return result
