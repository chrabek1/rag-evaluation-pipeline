from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoldenEvidence:
    text: str
    normalized_length: int
    
    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be empty")
        
        if self.normalized_length <= 0:
            raise ValueError(
                "normalized_length must be greater than 0"
            )