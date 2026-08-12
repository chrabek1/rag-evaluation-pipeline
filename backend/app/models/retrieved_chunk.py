from dataclasses import dataclass

from app.models.chunk import Chunk


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    chunk: Chunk
    score: float
    
    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be empty")