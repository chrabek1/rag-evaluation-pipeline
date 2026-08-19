from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidenceInterval:
    start: int
    end: int
    
    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must not be negative")
        
        if self.end <= self.start:
            raise ValueError(
                "end must be greater than start"
            )