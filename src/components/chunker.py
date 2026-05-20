from typing import List

class SemanticChunker:
    def __init__(self, llm_manager):
        """
        Initialisiert den semantischen Chunker mit einem LLM-Manager.
        
        :param llm_manager: LLMManager-Instanz für die semantische Segmentierung
        """
        self.llm_manager = llm_manager
    
    def chunk(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Teilt einen Text semantisch in Chunks auf.
        
        :param text: Eingabetext zum Aufteilen
        :param chunk_size: Zielgröße für die Chunks
        :return: Liste von Text-Chunks
        """
        # In einer produktiven Umgebung würde man hier ein komplexeres
        # semantisches Chunking mit dem LLM implementieren
        
        # Für dieses Demo-System verwenden wir eine einfache Aufteilung
        # bei doppelten Zeilenumbrüchen (Absätze)
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Wenn der aktuelle Chunk mit dem neuen Absatz zu lang wäre, speichern wir ihn
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # Ansonsten fügen wir den Absatz zum aktuellen Chunk hinzu
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Fügen Sie den letzten Chunk hinzu, falls er nicht leer ist
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Bei sehr kurzen Texten geben wir den Originaltext zurück
        if not chunks and text:
            chunks = [text]
            
        return chunks