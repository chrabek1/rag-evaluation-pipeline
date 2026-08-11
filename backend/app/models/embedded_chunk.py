from dataclasses import dataclass

from app.models.chunk import Chunk


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk_id: str
    chunk: Chunk
    embedding: list[float]
    
    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id cannot be empty")
        
        if not self.embedding:
            raise ValueError("embedding cannot be empty")