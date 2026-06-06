import logging
import os
from dataclasses import dataclass
from typing import Optional

import dspy
from dotenv import load_dotenv

from src.config.config_manager import ConfigManager
from src.components.chunker import SemanticChunker
from src.components.document_processor import DocumentProcessor
from src.components.embedding_manager import EmbeddingManager
from src.components.hyde_rag import HyDERAG
from src.components.lancedb_manager import LanceDBManager
from src.components.llm_manager import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class Components:
    config_manager: ConfigManager
    chunk_llm_manager: LLMManager
    answer_llm_manager: LLMManager
    embedding_manager: EmbeddingManager
    db_manager: LanceDBManager
    chunker: SemanticChunker
    document_processor: DocumentProcessor
    hyde_rag: HyDERAG

    @property
    def initialised(self) -> bool:
        return all(
            getattr(self, field) is not None
            for field in (
                "config_manager",
                "chunk_llm_manager",
                "answer_llm_manager",
                "embedding_manager",
                "db_manager",
                "chunker",
                "document_processor",
                "hyde_rag",
            )
        )


def _resolve_config_path(explicit: Optional[str] = None) -> str:
    """
    Resolves settings.yaml in this order:
    1. explicit path
    2. $HYDSPY_CONFIG (environment variable)
    3. src/config/settings.yaml (relative to the project root)
    """
    if explicit:
        return explicit
    env_path = os.getenv("HYDSPY_CONFIG")
    if env_path:
        return env_path
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "config",
        "settings.yaml",
    )


def build(config_path: Optional[str] = None, load_env: bool = True) -> Components:
    """
    Initialises the full component chain.

    :param config_path: Optional path to settings.yaml
    :param load_env: Load .env automatically
    :return: Components dataclass with all managers initialised
    :raises RuntimeError: If the DSPy LM initialisation fails
    """
    if load_env:
        load_dotenv()

    config_manager = ConfigManager(config_path=_resolve_config_path(config_path))
    settings = config_manager.get_settings()
    chunk_size = int(settings.get("chunk_size", 1000))

    chunk_llm_config = config_manager.get_llm_config("chunk_llm")
    answer_llm_config = config_manager.get_llm_config("answer_llm")
    embedding_config = config_manager.get_embedding_config()
    lancedb_config = config_manager.get_lancedb_config()

    chunk_llm_manager = LLMManager(
        base_url=chunk_llm_config["base_url"],
        api_key=chunk_llm_config["api_key"],
        model_id=chunk_llm_config["model_id"],
    )

    answer_llm_manager = LLMManager(
        base_url=answer_llm_config["base_url"],
        api_key=answer_llm_config["api_key"],
        model_id=answer_llm_config["model_id"],
    )

    embedding_manager = EmbeddingManager(
        base_url=embedding_config["base_url"],
        api_key=embedding_config["api_key"],
        model_id=embedding_config["model_id"],
    )

    db_manager = LanceDBManager(
        uri=lancedb_config["uri"],
        table_name=lancedb_config["table_name"],
    )

    # DSPy configuration. We fail loud if the LLM configuration is invalid.
    answer_lm = dspy.LM(
        model=answer_llm_config["model_id"],
        api_key=answer_llm_config["api_key"],
        base_url=answer_llm_config["base_url"],
    )
    dspy.configure(lm=answer_lm)
    logger.info("DSPy configured with model '%s'.", answer_llm_config["model_id"])

    chunker = SemanticChunker(chunk_llm_manager, chunk_size=chunk_size)
    document_processor = DocumentProcessor(chunker, embedding_manager, db_manager)
    hyde_rag = HyDERAG(db_manager, embedding_manager)

    return Components(
        config_manager=config_manager,
        chunk_llm_manager=chunk_llm_manager,
        answer_llm_manager=answer_llm_manager,
        embedding_manager=embedding_manager,
        db_manager=db_manager,
        chunker=chunker,
        document_processor=document_processor,
        hyde_rag=hyde_rag,
    )
