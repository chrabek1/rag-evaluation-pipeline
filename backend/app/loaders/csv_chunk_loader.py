import csv
from pathlib import Path

from app.models.chunk import Chunk


class CsvChunkLoader:
    REQUIRED_COLUMNS = {"filename", "content"}
    def load(self, path: Path) -> list[Chunk]:
        chunks = []
        
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            
            if reader.fieldnames is None or not self.REQUIRED_COLUMNS.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    "CSV must contain columns: filename, content"
                )
            
            for row in reader:
                chunks.append(
                    Chunk(
                        filename=row["filename"],
                        content=row["content"],
                    )
                )
    
        return chunks