import os
import sys
from dotenv import load_dotenv

# Füge das src-Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config.config_manager import ConfigManager
from src.components.llm_manager import LLMManager
from src.components.embedding_manager import EmbeddingManager
from src.components.lancedb_manager import LanceDBManager
from src.components.chunker import SemanticChunker
from src.components.document_processor import DocumentProcessor
from src.components.hyde_rag import HyDERAG

# DSPy importieren
import dspy

def main():
    # Lade Umgebungsvariablen
    load_dotenv()
    
    # Initialisiere Konfigurationsmanager
    config_manager = ConfigManager()
    
    # Hole Konfigurationen
    chunk_llm_config = config_manager.get_llm_config("chunk_llm")
    answer_llm_config = config_manager.get_llm_config("answer_llm")
    embedding_config = config_manager.get_embedding_config()
    lancedb_config = config_manager.get_lancedb_config()
    
    # Initialisiere Komponenten
    chunk_llm_manager = LLMManager(
        base_url=chunk_llm_config["base_url"],
        api_key=chunk_llm_config["api_key"],
        model_id=chunk_llm_config["model_id"]
    )
    
    answer_llm_manager = LLMManager(
        base_url=answer_llm_config["base_url"],
        api_key=answer_llm_config["api_key"],
        model_id=answer_llm_config["model_id"]
    )
    
    embedding_manager = EmbeddingManager(
        base_url=embedding_config["base_url"],
        api_key=embedding_config["api_key"],
        model_id=embedding_config["model_id"]
    )
    
    db_manager = LanceDBManager(
        uri=lancedb_config["uri"],
        table_name=lancedb_config["table_name"]
    )
    
    # Initialisiere DSPy mit dem Antwort-LLM
    try:
        answer_lm = dspy.LM(
            model=answer_llm_config["model_id"],
            api_key=answer_llm_config["api_key"],
            base_url=answer_llm_config["base_url"]
        )
        dspy.configure(lm=answer_lm)
    except Exception as e:
        print(f"Konnte DSPy LLM nicht mit echter Konfiguration initialisieren: {e}")
        print("Verwende Mock-Konfiguration für Demo-Zwecke")
        # Fallback zu Mock für Demo-Zwecke
        answer_lm = dspy.LM(model="mock")
        dspy.configure(lm=answer_lm)
    
    # Initialisiere Chunker und DocumentProcessor
    chunker = SemanticChunker(chunk_llm_manager)
    document_processor = DocumentProcessor(chunker, embedding_manager, db_manager)
    
    # Initialisiere HyDE-RAG
    hyde_rag = HyDERAG(db_manager, embedding_manager)
    
    # Beispiel für die Verarbeitung eines Dokuments
    sample_document = """
    Die Kombination aus Hypothetical Document Embeddings (HyDE) und DSPy ist in modernen 
    Retrieval-Augmented Generation (RAG) Systemen sehr effektiv. HyDE verbessert die 
    semantische Suche, indem es hypothetische Dokumente generiert, die der eigentlichen 
    Anfrage ähnlicher sind als die ursprüngliche Frage. DSPy optimiert die Prompts und 
    die gesamte Pipeline, um bessere Ergebnisse zu erzielen.
    
    HyDE arbeitet wie folgt: Wenn ein Nutzer eine Frage stellt, generiert HyDE zunächst
    ein hypothetisches Dokument, das die Antwort auf diese Frage enthalten könnte. Dieses
    hypothetische Dokument wird dann in der Vektordatenbank gesucht, was bessere Treffer
    als die ursprüngliche Frage liefert.
    
    DSPy optimiert dann die gesamte Pipeline durch automatisches Prompt-Engineering.
    Es lernt aus Beispieldaten, welche Prompts für die beste Leistung sorgen, und passt
    sie automatisch an.
    """
    
    # Verarbeite das Dokument
    print("Verarbeite Dokument...")
    processed_chunks = document_processor.process_document("sample_doc_1", sample_document)
    print(f"Dokument verarbeitet. {len(processed_chunks)} Chunks erstellt.")
    
    # Beispiel für eine Frage
    question = "Wie funktioniert HyDE mit DSPy?"
    
    # Führe HyDE-RAG aus
    print(f"\nBeantworte Frage: {question}")
    try:
        result = hyde_rag(question)  # Verwende den korrekten Aufruf
        print(f"Antwort: {result.answer}")
        print(f"Kontext: {result.context}")
    except Exception as e:
        print(f"Fehler bei der Fragebeantwortung: {str(e)}")
        # Fallback für Demo
        print("\nFallback-Antwort:")
        print("Antwort: HyDE generiert hypothetische Dokumente zur Verbesserung der semantischen Suche, während DSPy die Prompts und die gesamte Pipeline optimiert.")
        print("Kontext: Die Kombination beider Technologien führt zu präziseren Retrieval-Ergebnissen in RAG-Systemen.")

if __name__ == "__main__":
    main()