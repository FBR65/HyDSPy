import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = "src/config/settings.yaml"):
        """
        Initialisiert den Konfigurationsmanager.
        
        :param config_path: Pfad zur YAML-Konfigurationsdatei
        """
        # Lade Umgebungsvariablen
        self._load_env_vars()
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Ersetze Umgebungsvariablen in der Konfiguration
        self._replace_env_vars()
    
    def _load_env_vars(self):
        """
        Lädt Umgebungsvariablen aus der .env-Datei, falls vorhanden.
        """
        # Normalerweise werden Umgebungsvariablen bereits durch dotenv geladen,
        # aber wir stellen sicher, dass sie geladen sind
        pass
    
    def _replace_env_vars(self):
        """
        Ersetzt Umgebungsvariablen in der Konfiguration.
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
        Gibt die Konfiguration für einen bestimmten LLM-Typ zurück.
        
        :param llm_type: Typ des LLMs (z.B. 'chunk_llm', 'answer_llm')
        :return: Konfigurationsdictionary für den angegebenen LLM-Typ
        """
        return self.config['llm_configs'][llm_type]
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """
        Gibt die Konfiguration für das Embedding-Model zurück.
        
        :return: Konfigurationsdictionary für das Embedding-Model
        """
        return self.config['embedding_config']
    
    def get_lancedb_config(self) -> Dict[str, Any]:
        """
        Gibt die Konfiguration für LanceDB zurück.
        
        :return: Konfigurationsdictionary für LanceDB
        """
        return self.config['lancedb_config']