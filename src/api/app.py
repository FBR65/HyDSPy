import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Füge das src-Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Lade Umgebungsvariablen
from dotenv import load_dotenv
load_dotenv()

# Importiere Komponenten nach Umgebungsvariablen-Loading
try:
    from src.config.config_manager import ConfigManager
    from src.components.llm_manager import LLMManager
    from src.components.embedding_manager import EmbeddingManager
    from src.components.lancedb_manager import LanceDBManager
    from src.components.chunker import SemanticChunker
    from src.components.document_processor import DocumentProcessor
    from src.components.hyde_rag import HyDERAG
    import dspy
    IMPORT_SUCCESS = True
except ImportError as e:
    logger.error(f"Import-Fehler: {e}")
    IMPORT_SUCCESS = False

app = FastAPI(title="HyDE-RAG API", description="API für das HyDE-RAG System")

# Globale Variablen für die Komponenten
components = {
    'config_manager': None,
    'chunk_llm_manager': None,
    'answer_llm_manager': None,
    'embedding_manager': None,
    'db_manager': None,
    'chunker': None,
    'document_processor': None,
    'hyde_rag': None
}

class Document(BaseModel):
    id: str
    text: str

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    context: str

@app.on_event("startup")
async def startup_event():
    global components
    if not IMPORT_SUCCESS:
        logger.error("API konnte nicht gestartet werden wegen fehlender Abhängigkeiten")
        return
    
    try:
        logger.info("Initialisiere API-Komponenten...")
        
        # Initialisiere Konfigurationsmanager
        components['config_manager'] = ConfigManager()
        
        # Hole Konfigurationen
        chunk_llm_config = components['config_manager'].get_llm_config("chunk_llm")
        answer_llm_config = components['config_manager'].get_llm_config("answer_llm")
        embedding_config = components['config_manager'].get_embedding_config()
        lancedb_config = components['config_manager'].get_lancedb_config()
        
        # Initialisiere Komponenten
        components['chunk_llm_manager'] = LLMManager(
            base_url=chunk_llm_config["base_url"],
            api_key=chunk_llm_config["api_key"],
            model_id=chunk_llm_config["model_id"]
        )
        
        components['answer_llm_manager'] = LLMManager(
            base_url=answer_llm_config["base_url"],
            api_key=answer_llm_config["api_key"],
            model_id=answer_llm_config["model_id"]
        )
        
        components['embedding_manager'] = EmbeddingManager(
            base_url=embedding_config["base_url"],
            api_key=embedding_config["api_key"],
            model_id=embedding_config["model_id"]
        )
        
        components['db_manager'] = LanceDBManager(
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
            logger.info("DSPy erfolgreich konfiguriert")
        except Exception as e:
            logger.warning(f"DSPy-Konfiguration fehlgeschlagen, verwende Mock: {e}")
            answer_lm = dspy.LM(model="mock")
            dspy.configure(lm=answer_lm)
        
        # Initialisiere Chunker und DocumentProcessor
        components['chunker'] = SemanticChunker(components['chunk_llm_manager'])
        components['document_processor'] = DocumentProcessor(
            components['chunker'], 
            components['embedding_manager'], 
            components['db_manager']
        )
        
        # Initialisiere HyDE-RAG
        components['hyde_rag'] = HyDERAG(
            components['db_manager'], 
            components['embedding_manager']
        )
        
        logger.info("API-Komponenten erfolgreich initialisiert")
    except Exception as e:
        logger.error(f"Fehler bei der API-Initialisierung: {e}")
        # Setze alle Komponenten auf None bei Fehler
        for key in components:
            components[key] = None

@app.post("/process_document", response_model=List[dict])
async def process_document(document: Document):
    """
    Verarbeitet ein Dokument durch Chunking, Vektorisierung und speichert es in der Datenbank.
    """
    if components['document_processor'] is None:
        raise HTTPException(status_code=500, detail="Dokumentenprozessor nicht initialisiert")
    
    try:
        processed_chunks = components['document_processor'].process_document(document.id, document.text)
        return processed_chunks
    except Exception as e:
        logger.error(f"Fehler bei der Dokumentenverarbeitung: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler bei der Dokumentenverarbeitung: {str(e)}")

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """
    Beantwortet eine Frage basierend auf den in der Datenbank gespeicherten Dokumenten.
    """
    if components['hyde_rag'] is None:
        raise HTTPException(status_code=500, detail="HyDE-RAG nicht initialisiert")
    
    try:
        result = components['hyde_rag'](question.question)
        return Answer(answer=result.answer, context=result.context)
    except Exception as e:
        logger.error(f"Fehler bei der Fragebeantwortung: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler bei der Fragebeantwortung: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Prüft den Status der API.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)