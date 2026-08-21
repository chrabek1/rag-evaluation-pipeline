from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be empty")

        if not self.model.strip():
            raise ValueError("model must not be empty")

        if self.input_tokens is not None and self.input_tokens < 0:
            raise ValueError("input_tokens must not be negative")

        if self.output_tokens is not None and self.output_tokens < 0:
            raise ValueError("output_tokens must not be negative")

    @property
    def total_tokens(self) -> int | None:
        if (
            self.input_tokens is None
            or self.output_tokens is None
        ):
            return None

        return self.input_tokens + self.output_tokens