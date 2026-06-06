"""Tests for src.components.embedding_manager.EmbeddingManager."""
import pytest
from openai import APIConnectionError, OpenAIError

from src.components.embedding_manager import EmbeddingManager


class _FakeEmbeddings:
    """Stub response that mimics an OpenAI embeddings response."""

    def __init__(self, data):
        self.data = data


class _FakeEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeClient:
    """Stub client for embeddings.create()."""

    def __init__(self, embeddings_to_return=None, raise_exc=None):
        self._embeddings = (
            _FakeEmbeddings([_FakeEmbeddingItem(e) for e in embeddings_to_return])
            if embeddings_to_return is not None
            else None
        )
        self.raise_exc = raise_exc
        self.calls: list[dict] = []
        self.embeddings = self._EmbeddingsNamespace(self)

    class _EmbeddingsNamespace:
        def __init__(self, parent):
            self._parent = parent

        def create(self, *, input, model):
            self._parent.calls.append({"input": input, "model": model})
            if self._parent.raise_exc is not None:
                raise self._parent.raise_exc
            return self._parent._embeddings


def test_returns_embeddings_and_caches_dimension(monkeypatch):
    fake = _FakeClient(embeddings_to_return=[[0.1] * 8, [0.2] * 8])
    em = EmbeddingManager("http://x", "k", "m")
    monkeypatch.setattr(em, "client", fake)

    result = em.get_embeddings(["a", "b"])
    assert result == [[0.1] * 8, [0.2] * 8]
    assert em.dimension == 8
    assert fake.calls[0]["input"] == ["a", "b"]
    assert fake.calls[0]["model"] == "m"


def test_api_error_raises_runtime_error(monkeypatch):
    fake = _FakeClient(raise_exc=APIConnectionError(request=None))
    em = EmbeddingManager("http://x", "k", "m")
    monkeypatch.setattr(em, "client", fake)
    with pytest.raises(RuntimeError, match="Embedding creation"):
        em.get_embeddings(["x"])


def test_dimension_property_raises_before_first_call():
    em = EmbeddingManager("http://x", "k", "m")
    with pytest.raises(RuntimeError, match="dimension"):
        _ = em.dimension


def test_empty_input_returns_empty_list(monkeypatch):
    fake = _FakeClient(embeddings_to_return=[])
    em = EmbeddingManager("http://x", "k", "m")
    monkeypatch.setattr(em, "client", fake)
    assert em.get_embeddings([]) == []


def test_openai_error_is_wrapped(monkeypatch):
    """Generic OpenAI errors must surface as RuntimeError."""
    fake = _FakeClient(raise_exc=OpenAIError("rate limit"))
    em = EmbeddingManager("http://x", "k", "m")
    monkeypatch.setattr(em, "client", fake)
    with pytest.raises(RuntimeError, match="Embedding creation"):
        em.get_embeddings(["x"])
