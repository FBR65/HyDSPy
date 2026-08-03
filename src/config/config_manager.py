import os
import re
from typing import Any, Dict

import yaml

# Matches ${VAR_NAME} placeholders anywhere inside a string value.
# Supports ASCII env-var names (letters, digits, underscore) and is applied
# repeatedly so a value like "Bearer ${TOKEN}" is substituted correctly.
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigManager:
    def __init__(self, config_path: str = "src/config/settings.yaml"):
        """
        Initialises the configuration manager.

        :param config_path: Path to the YAML configuration file
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.config = self._replace_env_vars(self.config)

    def _replace_env_vars(self, value: Any) -> Any:
        """
        Replaces ``${ENV_VAR}`` placeholders with the corresponding
        ``os.environ`` value.

        Placeholders may appear anywhere in the string (e.g. ``"Bearer
        ${TOKEN}"``) and are substituted repeatedly. Missing env vars are
        left as the literal placeholder string. Handles nested dicts and
        lists recursively.

        :param value: A value from the YAML config (str, dict, list, or other)
        :return: The value with env vars substituted (same type)
        """

        def _sub(match: re.Match) -> str:
            env_var_name = match.group(1)
            return os.getenv(env_var_name, match.group(0))

        if isinstance(value, dict):
            return {k: self._replace_env_vars(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._replace_env_vars(v) for v in value]
        if isinstance(value, str):
            return _ENV_VAR_RE.sub(_sub, value)
        return value

    def get_llm_config(self, llm_type: str) -> Dict[str, Any]:
        """
        Returns the configuration for a given LLM type.

        :param llm_type: Type of LLM (e.g. 'chunk_llm', 'answer_llm')
        :return: Configuration dictionary for the given LLM type
        """
        return self.config["llm_configs"][llm_type]

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Returns the configuration for the embedding model.

        :return: Configuration dictionary for the embedding model
        """
        return self.config["embedding_config"]

    def get_lancedb_config(self) -> Dict[str, Any]:
        """
        Returns the configuration for LanceDB.

        :return: Configuration dictionary for LanceDB
        """
        return self.config["lancedb_config"]

    def get_settings(self) -> Dict[str, Any]:
        """
        Returns the general settings block (chunk_size, etc.).

        :return: Dictionary with general settings; defaults to
            {'chunk_size': 1000} when the block is absent.
        """
        return self.config.get("settings", {"chunk_size": 1000})
