"""Tests for src.components.llm_manager.LLMManager."""

import pytest
from openai import APIConnectionError

from src.components.llm_manager import LLMManager


class _FakeChoice:
    def __init__(self, text=None, content=None):
        self.text = text
        self.message = _FakeMessage(content)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeCompletionsResponse:
    def __init__(self, text):
        self.choices = [_FakeChoice(text=text)]


class _FakeChatResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content=content)]


class _FakeClient:
    def __init__(self, completion_text="", chat_content="", raise_exc=None):
        self.completion_text = completion_text
        self.chat_content = chat_content
        self.raise_exc = raise_exc
        self.completion_calls: list[dict] = []
        self.chat_calls: list[dict] = []

    class _Completions:
        def __init__(self, parent):
            self._parent = parent

        def create(self, *, model, prompt, max_tokens):
            self._parent.completion_calls.append(
                {"model": model, "prompt": prompt, "max_tokens": max_tokens}
            )
            if self._parent.raise_exc is not None:
                raise self._parent.raise_exc
            return _FakeCompletionsResponse(self._parent.completion_text)

    @property
    def completions(self):
        return self._Completions(self)

    class _ChatCompletions:
        def __init__(self, parent):
            self._parent = parent

        def create(self, *, model, messages, max_tokens):
            self._parent.chat_calls.append(
                {"model": model, "messages": messages, "max_tokens": max_tokens}
            )
            if self._parent.raise_exc is not None:
                raise self._parent.raise_exc
            return _FakeChatResponse(self._parent.chat_content)

    class _ChatNamespace:
        def __init__(self, parent):
            self._parent = parent
            self.completions = parent._ChatCompletions(parent)

    @property
    def chat(self):
        return self._ChatNamespace(self)


def test_generate_completion_raises_on_api_error(monkeypatch):
    fake = _FakeClient(raise_exc=APIConnectionError(request=None))
    lm = LLMManager("http://x", "k", "m")
    monkeypatch.setattr(lm, "client", fake)
    with pytest.raises(RuntimeError, match="LLM completion"):
        lm.generate_completion("hi")


def test_generate_completion_returns_stripped_text(monkeypatch):
    # generate_completion now uses the chat-completions endpoint, so the
    # response content comes back via the chat path.
    fake = _FakeClient(chat_content="  hello world  ")
    lm = LLMManager("http://x", "k", "m")
    monkeypatch.setattr(lm, "client", fake)
    assert lm.generate_completion("hi") == "hello world"
    # The prompt is sent as a single user message.
    assert fake.chat_calls[0]["model"] == "m"
    assert fake.chat_calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert fake.chat_calls[0]["max_tokens"] == 500


def test_generate_chat_raises_on_api_error(monkeypatch):
    fake = _FakeClient(raise_exc=APIConnectionError(request=None))
    lm = LLMManager("http://x", "k", "m")
    monkeypatch.setattr(lm, "client", fake)
    with pytest.raises(RuntimeError, match="LLM chat completion"):
        lm.generate_chat_completion([{"role": "user", "content": "hi"}])


def test_generate_chat_returns_stripped_content(monkeypatch):
    fake = _FakeClient(chat_content="  response  ")
    lm = LLMManager("http://x", "k", "m")
    monkeypatch.setattr(lm, "client", fake)
    msgs = [{"role": "user", "content": "hi"}]
    assert lm.generate_chat_completion(msgs) == "response"
    assert fake.chat_calls[0]["messages"] == msgs


def test_generate_completion_custom_max_tokens(monkeypatch):
    fake = _FakeClient(chat_content="x")
    lm = LLMManager("http://x", "k", "m")
    monkeypatch.setattr(lm, "client", fake)
    lm.generate_completion("hi", max_tokens=100)
    assert fake.chat_calls[0]["max_tokens"] == 100


def test_generate_completion_none_content_returns_empty(monkeypatch):
    """When the model returns content=None, the method must not crash on
    .strip() and must return an empty string."""
    fake = _FakeClient(chat_content=None)
    lm = LLMManager("http://x", "k", "m")
    monkeypatch.setattr(lm, "client", fake)
    assert lm.generate_completion("hi") == ""


def test_generate_chat_none_content_returns_empty(monkeypatch):
    """When chat content is None, generate_chat_completion must not crash."""
    fake = _FakeClient(chat_content=None)
    lm = LLMManager("http://x", "k", "m")
    monkeypatch.setattr(lm, "client", fake)
    assert lm.generate_chat_completion([{"role": "user", "content": "hi"}]) == ""
