#!/usr/bin/env python3
"""
Direktes API-Testskript für das HyDE-RAG-System

Dieses Skript testet die API-Funktionen direkt ohne HTTP-Server.
"""

import os
import sys
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

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

def initialize_components():
    """Initialisiert alle Komponenten des Systems."""
    try:
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
        except Exception:
            # Falls die Konfiguration fehlschlägt, verwenden wir einen Mock
            answer_lm = dspy.LM(model="mock")
            dspy.configure(lm=answer_lm)
        
        # Initialisiere Chunker und DocumentProcessor
        chunker = SemanticChunker(chunk_llm_manager)
        document_processor = DocumentProcessor(chunker, embedding_manager, db_manager)
        
        # Initialisiere HyDE-RAG
        hyde_rag = HyDERAG(db_manager, embedding_manager)
        
        return {
            'document_processor': document_processor,
            'hyde_rag': hyde_rag
        }
    except Exception as e:
        print(f"Fehler bei der Initialisierung: {e}")
        return None

def upload_document(components, doc_id: str, text: str):
    """Lädt ein Dokument in das System hoch."""
    try:
        processed_chunks = components['document_processor'].process_document(doc_id, text)
        print(f"Dokument '{doc_id}' erfolgreich hochgeladen.")
        print(f"Verarbeitete Chunks: {len(processed_chunks)}")
        return True
    except Exception as e:
        print(f"Fehler beim Hochladen des Dokuments: {e}")
        return False

def ask_question(components, question: str):
    """Stellt eine Frage an das System."""
    try:
        result = components['hyde_rag'](question)
        print(f"\nFrage: {question}")
        print(f"Antwort: {result.answer}")
        print(f"Kontext: {result.context}")
        return True
    except Exception as e:
        print(f"Fehler bei der Fragebeantwortung: {e}")
        return False

def main():
    """Hauptfunktion für den API-Test."""
    print("HyDE-RAG API-Test (Direkter Modus)")
    print("=" * 50)
    
    # Initialisiere Komponenten
    print("Initialisiere Komponenten...")
    components = initialize_components()
    
    if components is None:
        print("Konnte Komponenten nicht initialisieren.")
        return
    
    # Beispiel-Dokument
    sample_document = """
    Maschinelles Lernen ist ein Teilbereich der künstlichen Intelligenz, der es Systemen 
    ermöglicht, aus Daten zu lernen und Vorhersagen oder Entscheidungen zu treffen, 
    ohne explizit programmiert zu werden. Es gibt drei Hauptkategorien des maschinellen 
    Lernens:
    
    1. Überwachtes Lernen (Supervised Learning): Verwendet gelabelte Trainingsdaten, 
       um Muster zu erkennen und Vorhersagen zu treffen. Beispiele sind Klassifikation 
       und Regression.
    
    2. Unüberwachtes Lernen (Unsupervised Learning): Findet Muster in ungelabelten 
       Daten. Clustering und Assoziationsregel-Lernen sind gängige Anwendungen.
    
    3. Verstärkendes Lernen (Reinforcement Learning): Lernt durch Interaktion mit 
       einer Umgebung, um Belohnungen zu maximieren. Wird häufig in Robotik und 
       Spieltheorie eingesetzt.
    
    Moderne Anwendungen von maschinellem Lernen umfassen Bild- und Spracherkennung, 
    Empfehlungssysteme, autonome Fahrzeuge und medizinische Diagnose.
    """
    
    # Dokument hochladen
    print("\n1. Dokument hochladen...")
    if upload_document(components, "ml_intro_1", sample_document):
        print("Dokument erfolgreich verarbeitet.")
    else:
        print("Dokument konnte nicht hochgeladen werden.")
        return
    
    # Frage stellen
    print("\n2. Frage stellen...")
    question = "Welche Arten des maschinellen Lernens gibt es?"
    ask_question(components, question)
    
    # Zweite Frage
    print("\n3. Zweite Frage...")
    question2 = "Wie funktioniert überwachtes Lernen?"
    ask_question(components, question2)

if __name__ == "__main__":
    main()