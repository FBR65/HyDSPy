#!/usr/bin/env python3
"""
FINALES API-Testskript für das HyDE-RAG-System

Verwendung:
1. Starten Sie den API-Server in einem separaten Terminal:
   uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000

2. Führen Sie dieses Skript aus:
   uv run python final_api_test.py
"""

import requests
import json
import time

# API-Konfiguration
BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"Content-Type": "application/json"}

def check_api_health():
    """Prüft, ob die API erreichbar ist."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_document(doc_id: str, text: str):
    """Lädt ein Dokument in das System hoch."""
    url = f"{BASE_URL}/process_document"
    payload = {"id": doc_id, "text": text}
    
    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=30)
        return response.status_code == 200
    except:
        return False

def ask_question(question: str):
    """Stellt eine Frage an das System."""
    url = f"{BASE_URL}/ask"
    payload = {"question": question}
    
    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def main():
    """Hauptfunktion für den API-Test."""
    print("📡 HyDE-RAG API-Test")
    print("=" * 50)
    
    # Warte auf API-Verfügbarkeit
    print("⏳ Warte auf API-Server...")
    for i in range(10):
        if check_api_health():
            print("✅ API ist bereit!")
            break
        time.sleep(1)
    else:
        print("❌ API nicht verfügbar. Bitte starten Sie den Server:")
        print("   uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000")
        return
    
    # Testdokument
    document = """
    HyDE (Hypothetical Document Embeddings) und DSPy sind Schlüsseltechnologien 
    in modernen RAG-Systemen. HyDE verbessert die Retrieval-Qualität durch 
    Generierung hypothetischer Dokumente. DSPy optimiert die gesamte Pipeline 
    durch automatisches Prompt-Engineering.
    """
    
    # Upload Test
    print("\n📤 Lade Dokument hoch...")
    if upload_document("test_doc", document):
        print("✅ Dokument hochgeladen")
    else:
        print("❌ Fehler beim Hochladen")
        return
    
    time.sleep(1)
    
    # Frage Test
    print("\n❓ Frage stellen...")
    question = "Wie funktioniert HyDE mit DSPy?"
    result = ask_question(question)
    
    if result:
        print(f"Frage: {question}")
        print(f"Antwort: {result['answer']}")
        print(f"Kontext: {result['context'][:100]}...")
        print("✅ API-Test erfolgreich!")
    else:
        print("❌ Fehler bei der Fragebeantwortung")

if __name__ == "__main__":
    main()