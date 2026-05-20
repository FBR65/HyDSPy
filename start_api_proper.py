#!/usr/bin/env python3
"""
Eigenständiges Startskript für die HyDE-RAG API
"""

import os
import sys
import subprocess
import time

def start_api_server():
    """Startet den API-Server."""
    print("🚀 Starte HyDE-RAG API-Server...")
    print("   Server wird unter http://127.0.0.1:8000 verfügbar sein")
    print("   Beenden mit STRG+C\n")
    
    # Wechsle ins Projektverzeichnis
    os.chdir("/home/speedy/Dokumente/dev/HyDSPy")
    
    # Starte den Server
    try:
        process = subprocess.Popen([
            "uv", "run", "uvicorn", "src.api.app:app", 
            "--host", "127.0.0.1", 
            "--port", "8000"
        ])
        
        print(f"✅ API-Server gestartet (PID: {process.pid})")
        print("💡 API ist nun bereit für Anfragen")
        return process
    except Exception as e:
        print(f"❌ Fehler beim Starten des Servers: {e}")
        return None

if __name__ == "__main__":
    process = start_api_server()
    if process:
        try:
            # Warte auf das Beenden des Prozesses
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Beende API-Server...")
            process.terminate()
            process.wait()
            print("✅ API-Server beendet")