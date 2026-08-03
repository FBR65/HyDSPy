"""Tests for src.components.chunker.SemanticChunker."""

import pytest

from src.components.chunker import SemanticChunker, _split_by_size, _split_words
from tests.conftest import ExplodingLLM, FakeLLM

# --- deterministic mode (without LLM) ---


def test_empty_text():
    chunker = SemanticChunker(chunk_size=100)
    assert chunker.chunk("") == []


def test_whitespace_only():
    chunker = SemanticChunker(chunk_size=100)
    assert chunker.chunk("   \n\n  ") == []


def test_short_text_returned_as_single_chunk():
    chunker = SemanticChunker(chunk_size=100)
    assert chunker.chunk("Hello world") == ["Hello world"]


def test_paragraphs_joined_when_under_limit():
    """Without LLM: paragraphs concatenated when they fit chunk_size."""
    chunker = SemanticChunker(chunk_size=1000)
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunker.chunk(text)
    assert chunks == ["Paragraph one.\n\nParagraph two.\n\nParagraph three."]


def test_paragraphs_split_when_over_limit():
    """Without LLM: when combined paragraphs exceed chunk_size, they are split."""
    chunker = SemanticChunker(chunk_size=20)
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunker.chunk(text)
    assert all(len(c) <= 20 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_chunk_size_respected_for_long_paragraphs():
    chunker = SemanticChunker(chunk_size=50)
    text = ("third " * 50).strip()
    chunks = chunker.chunk(text)
    assert all(len(c) <= 50 for c in chunks)
    assert len(chunks) > 1
    assert "".join(c.replace(" ", "") for c in chunks) == text.replace(" ", "")


def test_long_paragraph_with_sentences_is_split():
    chunker = SemanticChunker(chunk_size=50)
    text = (
        "First sentence. Second sentence. "
        "Third sentence with more text. Fourth sentence."
    )
    chunks = chunker.chunk(text)
    assert all(len(c) <= 50 for c in chunks)
    assert len(chunks) >= 2


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        SemanticChunker(chunk_size=0)
    with pytest.raises(ValueError):
        SemanticChunker(chunk_size=-1)


def test_override_chunk_size_per_call():
    chunker = SemanticChunker(chunk_size=1000)
    text = "A. B. C. D. E."
    chunks = chunker.chunk(text, chunk_size=3)
    assert all(len(c) <= 3 for c in chunks)


# --- LLM path ---


def test_llm_valid_json():
    llm = FakeLLM('{"chunks": ["Topic A", "Topic B"]}')
    chunker = SemanticChunker(llm_manager=llm, chunk_size=1000)
    chunks = chunker.chunk("any text")
    assert chunks == ["Topic A", "Topic B"]
    assert len(llm.calls) == 1


def test_llm_prompt_includes_chunk_size_guideline():
    """The prompt should communicate the target chunk size to the LLM."""
    llm = FakeLLM('{"chunks": []}')
    chunker = SemanticChunker(llm_manager=llm, chunk_size=750)
    chunker.chunk("text")
    assert "750" in llm.calls[0]


def test_llm_markdown_wrapped_json():
    llm = FakeLLM('```json\n{"chunks": ["a", "b"]}\n```')
    chunker = SemanticChunker(llm_manager=llm, chunk_size=1000)
    assert chunker.chunk("text") == ["a", "b"]


def test_llm_garbage_raises_value_error():
    chunker = SemanticChunker(llm_manager=FakeLLM("not json here"), chunk_size=1000)
    with pytest.raises(ValueError, match="not valid JSON"):
        chunker.chunk("Paragraph one.\n\nParagraph two.")


def test_llm_empty_response_raises_value_error():
    chunker = SemanticChunker(llm_manager=FakeLLM(""), chunk_size=1000)
    with pytest.raises(ValueError, match="empty"):
        chunker.chunk("Hello world")


def test_llm_runtime_error_propagates():
    chunker = SemanticChunker(llm_manager=ExplodingLLM(), chunk_size=1000)
    with pytest.raises(RuntimeError, match="simulated"):
        chunker.chunk("Hello world")


def test_llm_response_without_chunks_key_raises():
    llm = FakeLLM('{"other_key": ["a", "b"]}')
    chunker = SemanticChunker(llm_manager=llm, chunk_size=1000)
    with pytest.raises(ValueError, match="chunks"):
        chunker.chunk("Hello world")


def test_llm_response_with_non_list_chunks_raises():
    llm = FakeLLM('{"chunks": "not an array"}')
    chunker = SemanticChunker(llm_manager=llm, chunk_size=1000)
    with pytest.raises(ValueError, match="chunks"):
        chunker.chunk("Hello world")


def test_llm_response_with_non_string_items_raises():
    llm = FakeLLM('{"chunks": [1, 2, 3]}')
    chunker = SemanticChunker(llm_manager=llm, chunk_size=1000)
    with pytest.raises(ValueError, match="chunks"):
        chunker.chunk("Hello world")


# --- internal helpers ---


def test_split_words_handles_oversized_words():
    """Words longer than chunk_size must be split character-by-character."""
    chunks = _split_words("a" * 250, chunk_size=100)
    assert all(len(c) == 100 for c in chunks[:2])
    assert chunks[2] == "a" * 50


def test_split_words_handles_normal_words():
    """_split_words packs words until chunk_size is reached."""
    chunks = _split_words("alpha beta gamma delta", chunk_size=10)
    # 'alpha beta' (10) fits, 'gamma delta' (11) is too long
    # -> 'alpha beta', 'gamma', 'delta'
    assert chunks == ["alpha beta", "gamma", "delta"]
    assert all(len(c) <= 10 for c in chunks)


def test_split_by_size_short_text():
    assert _split_by_size("short text", 100) == ["short text"]


def test_split_by_size_paragraphs_joined_under_limit():
    """Multiple paragraphs are concatenated when they fit in chunk_size."""
    text = "First paragraph.\n\nSecond paragraph."
    chunks = _split_by_size(text, 100)
    assert chunks == ["First paragraph.\n\nSecond paragraph."]


def test_split_by_size_paragraphs_split_over_limit():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = _split_by_size(text, 15)
    assert all(len(c) <= 15 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_split_by_size_oversized_paragraph():
    text = ("x" * 200).strip()
    chunks = _split_by_size(text, 50)
    assert all(len(c) <= 50 for c in chunks)
    assert "".join(chunks) == text
