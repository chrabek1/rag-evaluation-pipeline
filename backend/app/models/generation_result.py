from dataclasses import dataclass

from app.models.llm_response import LLMResponse


@dataclass(frozen=True, slots=True)
class GenerationResult:
    response: LLMResponse
    latency_seconds: float

    def __post_init__(self) -> None:
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds must not be negative")

    @property
    def answer(self) -> str:
        return self.response.text