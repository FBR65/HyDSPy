"""Tests for src.bootstrap."""

import os
import sys

import pytest

# Make sure src.* imports work
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def test_resolve_config_path_explicit():
    from src.bootstrap import _resolve_config_path

    assert _resolve_config_path("/custom/path.yaml") == "/custom/path.yaml"


def test_resolve_config_path_env(monkeypatch):
    from src.bootstrap import _resolve_config_path

    monkeypatch.setenv("HYDSPY_CONFIG", "/from/env.yaml")
    assert _resolve_config_path() == "/from/env.yaml"


def test_resolve_config_path_default(monkeypatch, real_settings_path):
    from src.bootstrap import _resolve_config_path

    monkeypatch.delenv("HYDSPY_CONFIG", raising=False)
    assert _resolve_config_path() == real_settings_path


def test_build_returns_initialised_components(monkeypatch):
    """build() should work with the real settings.yaml and yield initialised components.

    dspy.LM is mocked to avoid any LiteLLM call.
    """
    import dspy
    from src import bootstrap

    monkeypatch.setattr(dspy, "LM", lambda **kwargs: object())
    monkeypatch.setattr(dspy, "configure", lambda lm: None)

    comps = bootstrap.build(load_env=False)
    assert comps.initialised
    assert comps.chunker.chunk_size == 1000  # default in real settings.yaml
