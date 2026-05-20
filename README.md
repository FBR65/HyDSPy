# HyDE-RAG System mit DSPy und LanceDB

Dieses Projekt implementiert ein fortschrittenes Retrieval-Augmented Generation (RAG) System unter Verwendung von Hypothetical Document Embeddings (HyDE) und DSPy. Das System ist modular und konfigurierbar, um verschiedene OpenAI-kompatible Endpunkte zu unterstützen.

## Funktionen

- **HyDE (Hypothetical Document Embeddings)**: Verbessert die semantische Suche durch Generierung hypothetischer Dokumente
- **DSPy**: Optimiert die Prompts und die gesamte Pipeline für bessere Ergebnisse
- **LanceDB**: Lokale Vektordatenbank für effiziente Ähnlichkeitssuche
- **Modulare Architektur**: Austauschbare Komponenten für LLMs, Embedding-Modelle und Datenbanken
- **RESTful API**: Einfache Schnittstelle für Dokumentenverarbeitung und Fragebeantwortung

## Voraussetzungen

- Python >= 3.9
- [uv](https://github.com/astral-sh/uv) Paketmanager

## Installation

1. Klone das Repository:
   ```bash
   git clone <repository-url>
   cd HyDE_DSPy
   ```

2. Erstelle eine virtuelle Umgebung und installiere die Abhängigkeiten:
   ```bash
   uv venv
   source .venv/bin/activate  # Unter Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

3. Erstelle eine `.env` Datei basierend auf `.env.example` und füge deine API-Schlüssel hinzu:
   ```bash
   cp .env.example .env
   # Bearbeite .env und füge deine API-Schlüssel hinzu
   ```

## Konfiguration

Die Konfiguration erfolgt über die Datei `src/config/settings.yaml`. Hier können verschiedene LLMs, Embedding-Modelle und die Datenbankkonfiguration festgelegt werden.

## Verwendung

### Über die Kommandozeile

Starte die Hauptanwendung:
```bash
python src/main.py
```

### Über die API

Starte den API-Server:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Oder führe das API-Skript direkt aus:
```bash
python src/api/app.py
```

#### API-Endpunkte

- `POST /process_document`: Verarbeitet ein Dokument und speichert es in der Datenbank
- `POST /ask`: Beantwortet eine Frage basierend auf den gespeicherten Dokumenten
- `GET /health`: Prüft den Status der API

### Beispiel-API-Nutzung

```bash
# Dokument verarbeiten
curl -X POST "http://localhost:8000/process_document" \
     -H "Content-Type: application/json" \
     -d '{"id": "doc1", "text": "Dein Dokumententext hier..."}'

# Frage stellen
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Deine Frage hier..."}'
```

## Projektstruktur

```
├── src/
│   ├── config/
│   │   ├── config_manager.py
│   │   └── settings.yaml
│   ├── components/
│   │   ├── chunker.py
│   │   ├── document_processor.py
│   │   ├── embedding_manager.py
│   │   ├── hyde_rag.py
│   │   ├── lancedb_manager.py
│   │   └── llm_manager.py
│   ├── api/
│   │   └── app.py
│   └── main.py
├── .env.example
├── .gitignore
├── HyDE_DSPy.md
├── plan.md
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Entwicklung

### Code-Qualität

Das Projekt verwendet Black für Code-Formatierung und Flake8 für Linting. Vor dem Commit sollten diese Tools ausgeführt werden:

```bash
black .
flake8 .
```

### Tests

Tests können mit pytest ausgeführt werden:

```bash
pytest
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert. Siehe die LICENSE-Datei für Details.