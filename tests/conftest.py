"""Shared test fixtures.

These fixtures are intentionally minimal: no network, no real LLMs.
External services (Ollama, embedding APIs) are replaced by stub clients
that the tests inject via monkeypatch.
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterator

import pytest

# Project root and src directory are used to locate on-disk fixtures such as
# the real settings.yaml. `src.*` imports work without a sys.path hack because
# the package is installed in editable mode (see pyproject.toml).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_PROJECT_ROOT, "src")


@pytest.fixture
def temp_lancedb_uri() -> Iterator[str]:
    """Liefert einen Pfad in einem temporären Verzeichnis für LanceDB."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def real_settings_path() -> str:
    """Pfad zur echten settings.yaml (für ConfigManager-Tests)."""
    return os.path.join(_SRC, "config", "settings.yaml")


class FakeLLM:
    """Stub-LLM, der fertig konfigurierte JSON-Antworten zurückgibt."""

    def __init__(self, response: str = '{"chunks": []}') -> None:
        self.response = response
        self.calls: list[str] = []

    def generate_completion(self, prompt: str, max_tokens: int = 500) -> str:
        self.calls.append(prompt)
        return self.response


class ExplodingLLM:
    """Stub-LLM, der bei jedem Aufruf RuntimeError wirft."""

    def generate_completion(self, prompt: str, max_tokens: int = 500) -> str:
        raise RuntimeError("simulated LLM failure")
