from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    filename: str
    content: str
    
    def __post_init__(self) -> None:
        if not self.filename.strip():
            raise ValueError("filename cannot be empty")
        
        if not self.content.strip():
            raise ValueError("content cannot be empty")