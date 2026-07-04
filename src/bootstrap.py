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
from src.components.bm25_manager import BM25Manager
from src.components.agentic_rag import AgenticRAG

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
    bm25_manager: BM25Manager
    agentic_rag: AgenticRAG

    @property
    def initialised(self) -> bool:
        """Best-effort None-check over all component fields.

        Note: ``build()`` is fail-loud — it raises on any initialisation
        failure rather than returning a partially-populated ``Components``.
        Therefore this property is effectively always ``True`` for an instance
        returned by ``build()``. It exists as a defensive guard for callers
        that construct ``Components`` manually (e.g. tests) and want to assert
        completeness without catching exceptions.
        """
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
                "bm25_manager",
                "agentic_rag",
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


def _resolve_cache_path(cache_path: Optional[str]) -> Optional[str]:
    """Resolves a relative BM25 cache path to an absolute one.

    Paths from ``settings.yaml`` are typically relative (e.g.
    ``./lancedb/bm25_cache.pkl``). Resolving them against the project root
    makes the cache location independent of the API server's working directory.
    Absolute paths and ``None`` are returned unchanged.
    """
    if not cache_path:
        return cache_path
    if os.path.isabs(cache_path):
        return cache_path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(project_root, cache_path))


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
    multi_query = bool(settings.get("multi_query", True))
    num_queries = int(settings.get("num_queries", 5))
    bm25_hybrid = bool(settings.get("bm25_hybrid", True))
    expansion_weight = float(settings.get("expansion_weight", 0.5))
    df_threshold = float(settings.get("df_threshold", 0.1))
    max_calls = int(settings.get("max_calls", 15))
    token_threshold = int(settings.get("token_threshold", 115_200))
    search_limit = int(settings.get("search_limit", 10))
    max_queries_per_call = int(settings.get("max_queries_per_call", 5))
    open_window = int(settings.get("open_window", 5))

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
        index_type=lancedb_config.get("index_type", "IVF_PQ"),
        num_partitions=lancedb_config.get("num_partitions"),
        num_sub_vectors=lancedb_config.get("num_sub_vectors"),
        index_train_threshold=int(lancedb_config.get("index_train_threshold", 256)),
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

    bm25_manager = BM25Manager(
        db_manager=db_manager,
        expansion_weight=expansion_weight,
        df_threshold=df_threshold,
        cache_path=_resolve_cache_path(lancedb_config.get("bm25_cache_path")),
    )
    hyde_rag = HyDERAG(
        db_manager=db_manager,
        embedding_manager=embedding_manager,
        bm25_manager=bm25_manager,
        multi_query=multi_query,
        num_queries=num_queries,
        bm25_hybrid=bm25_hybrid,
    )
    agentic_rag = AgenticRAG(
        db_manager=db_manager,
        embedding_manager=embedding_manager,
        bm25_manager=bm25_manager,
        max_calls=max_calls,
        token_threshold=token_threshold,
        search_limit=search_limit,
        max_queries_per_call=max_queries_per_call,
        open_window=open_window,
    )

    return Components(
        config_manager=config_manager,
        chunk_llm_manager=chunk_llm_manager,
        answer_llm_manager=answer_llm_manager,
        embedding_manager=embedding_manager,
        db_manager=db_manager,
        chunker=chunker,
        document_processor=document_processor,
        hyde_rag=hyde_rag,
        bm25_manager=bm25_manager,
        agentic_rag=agentic_rag,
    )
