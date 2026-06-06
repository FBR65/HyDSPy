"""Tests for src.config.config_manager.ConfigManager."""
import os
import textwrap

import pytest

from src.config.config_manager import ConfigManager


def test_loads_real_settings(real_settings_path):
    """The real settings.yaml must contain all expected sections."""
    cfg = ConfigManager(config_path=real_settings_path)
    assert "llm_configs" in cfg.config
    assert "embedding_config" in cfg.config
    assert "lancedb_config" in cfg.config
    assert "settings" in cfg.config


def test_env_var_substitution(monkeypatch, tmp_path):
    """${ENV_VAR} placeholders are replaced by os.environ values."""
    monkeypatch.setenv("MY_TEST_KEY", "sk-fake-123")
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            llm_configs:
              chunk_llm:
                base_url: http://x
                api_key: ${MY_TEST_KEY}
                model_id: m
            embedding_config:
              base_url: http://x
              api_key: ${MY_TEST_KEY}
              model_id: e
            lancedb_config:
              uri: ./lancedb
              table_name: docs
            settings:
              chunk_size: 1000
            """
        )
    )
    cfg = ConfigManager(config_path=str(yaml_path))
    assert cfg.get_llm_config("chunk_llm")["api_key"] == "sk-fake-123"
    assert cfg.get_embedding_config()["api_key"] == "sk-fake-123"


def test_missing_env_var_kept_as_placeholder(tmp_path):
    """When the env var is missing, the placeholder is left unchanged."""
    os.environ.pop("UNSET_TEST_KEY", None)
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            llm_configs:
              chunk_llm:
                base_url: http://x
                api_key: ${UNSET_TEST_KEY}
                model_id: m
            embedding_config:
              base_url: http://x
              api_key: k
              model_id: e
            lancedb_config:
              uri: ./lancedb
              table_name: docs
            """
        )
    )
    cfg = ConfigManager(config_path=str(yaml_path))
    assert cfg.get_llm_config("chunk_llm")["api_key"] == "${UNSET_TEST_KEY}"


def test_get_settings_default_when_missing(tmp_path):
    """Missing settings block yields sensible defaults."""
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            llm_configs:
              chunk_llm: {base_url: http://x, api_key: k, model_id: m}
            embedding_config: {base_url: http://x, api_key: k, model_id: e}
            lancedb_config: {uri: ./lancedb, table_name: docs}
            """
        )
    )
    cfg = ConfigManager(config_path=str(yaml_path))
    assert cfg.get_settings() == {"chunk_size": 1000}


def test_get_settings_override(tmp_path):
    """A present settings block is read correctly."""
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            llm_configs:
              chunk_llm: {base_url: http://x, api_key: k, model_id: m}
            embedding_config: {base_url: http://x, api_key: k, model_id: e}
            lancedb_config: {uri: ./lancedb, table_name: docs}
            settings:
              chunk_size: 500
            """
        )
    )
    cfg = ConfigManager(config_path=str(yaml_path))
    assert cfg.get_settings() == {"chunk_size": 500}


def test_get_llm_config_unknown_raises(tmp_path):
    """An unknown LLM section raises KeyError."""
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            llm_configs:
              chunk_llm: {base_url: http://x, api_key: k, model_id: m}
            embedding_config: {base_url: http://x, api_key: k, model_id: e}
            lancedb_config: {uri: ./lancedb, table_name: docs}
            """
        )
    )
    cfg = ConfigManager(config_path=str(yaml_path))
    with pytest.raises(KeyError):
        cfg.get_llm_config("nonexistent_llm")
