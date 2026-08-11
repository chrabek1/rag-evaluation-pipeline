import csv
from pathlib import Path

from app.models.chunk import Chunk


class CsvChunkLoader:
    def load(self, path: Path) -> list[Chunk]:
        chunks = []
        
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                chunks.append(
                    Chunk(
                        filename=row["filename"],
                        content=row["content"],
                    )
                )
    
        return chunks