import yaml
import os
from typing import Dict, Any


class ConfigManager:
    def __init__(self, config_path: str = "src/config/settings.yaml"):
        """
        Initialises the configuration manager.

        :param config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self._replace_env_vars()

    def _replace_env_vars(self):
        """
        Replaces ${ENV_VAR} placeholders in the configuration with values from os.environ.
        Missing env vars are left as the literal placeholder string.
        """
        def replace_in_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    replace_in_dict(value)
                elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var_name = value[2:-1]
                    d[key] = os.getenv(env_var_name, value)

        replace_in_dict(self.config)

    def get_llm_config(self, llm_type: str) -> Dict[str, Any]:
        """
        Returns the configuration for a given LLM type.

        :param llm_type: Type of LLM (e.g. 'chunk_llm', 'answer_llm')
        :return: Configuration dictionary for the given LLM type
        """
        return self.config['llm_configs'][llm_type]

    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Returns the configuration for the embedding model.

        :return: Configuration dictionary for the embedding model
        """
        return self.config['embedding_config']

    def get_lancedb_config(self) -> Dict[str, Any]:
        """
        Returns the configuration for LanceDB.

        :return: Configuration dictionary for LanceDB
        """
        return self.config['lancedb_config']

    def get_settings(self) -> Dict[str, Any]:
        """
        Returns the general settings block (chunk_size, etc.).

        :return: Dictionary with general settings; defaults to {'chunk_size': 1000}
            when the block is absent.
        """
        return self.config.get('settings', {'chunk_size': 1000})
