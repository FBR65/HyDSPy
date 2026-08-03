"""Tests for src.bootstrap."""

import os

from src.bootstrap import _resolve_cache_path, _resolve_config_path


def test_resolve_config_path_explicit():
    assert _resolve_config_path("/custom/path.yaml") == "/custom/path.yaml"


def test_resolve_config_path_env(monkeypatch):
    monkeypatch.setenv("HYDSPY_CONFIG", "/from/env.yaml")
    assert _resolve_config_path() == "/from/env.yaml"


def test_resolve_config_path_default(monkeypatch, real_settings_path):
    monkeypatch.delenv("HYDSPY_CONFIG", raising=False)
    assert _resolve_config_path() == real_settings_path


def test_resolve_cache_path_absolute_unchanged():
    assert _resolve_cache_path("/abs/path.pkl") == "/abs/path.pkl"


def test_resolve_cache_path_none_unchanged():
    assert _resolve_cache_path(None) is None


def test_resolve_cache_path_relative_made_absolute():
    """A relative cache path is resolved against the project root so it is
    stable regardless of the working directory."""
    result = _resolve_cache_path("relative/cache.pkl")
    assert result is not None
    assert os.path.isabs(result)
    assert result.endswith("relative/cache.pkl")


def test_build_returns_initialised_components(monkeypatch):
    """build() should work with the real settings.yaml and yield initialised
    components. dspy.LM is mocked to avoid any real API call.
    """
    import dspy

    from src import bootstrap

    monkeypatch.setattr(dspy, "LM", lambda **kwargs: object())
    monkeypatch.setattr(dspy, "configure", lambda lm: None)

    comps = bootstrap.build(load_env=False)
    assert comps.initialised
    assert comps.chunker.chunk_size == 1000  # default in real settings.yaml
    assert comps.agentic_rag is not None  # agentic_enabled defaults to True


def test_build_disables_agentic_when_disabled(monkeypatch, tmp_path):
    """When agentic_enabled=False, build() should leave agentic_rag as None
    while still initialising all other components.
    """
    import dspy

    from src import bootstrap

    monkeypatch.setattr(dspy, "LM", lambda **kwargs: object())
    monkeypatch.setattr(dspy, "configure", lambda lm: None)

    # Write a settings.yaml with agentic_enabled=False.
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text(
        """
llm_configs:
  chunk_llm: {base_url: http://x, api_key: k, model_id: m}
  answer_llm: {base_url: http://x, api_key: k, model_id: m}
embedding_config: {base_url: http://x, api_key: k, model_id: e}
lancedb_config: {uri: ./lancedb, table_name: docs}
settings:
  chunk_size: 1000
  agentic_enabled: false
"""
    )
    comps = bootstrap.build(config_path=str(yaml_path), load_env=False)
    assert comps.initialised
    assert comps.agentic_rag is None
    assert comps.hyde_rag is not None
