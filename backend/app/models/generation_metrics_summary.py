import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationMetricsSummary:
    query_count: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_latency_seconds: float
    total_input_tokens: int | None
    total_output_tokens: int | None

    def __post_init__(self) -> None:
        if self.query_count <= 0:
            raise ValueError("query_count must be greater than 0")

        self._validate_score(
            name="mean_faithfulness",
            value=self.mean_faithfulness,
            minimum=0.0,
            maximum=1.0,
        )
        self._validate_score(
            name="mean_answer_relevancy",
            value=self.mean_answer_relevancy,
            minimum=-1.0,
            maximum=1.0,
        )

        if (
            not math.isfinite(self.mean_latency_seconds)
            or self.mean_latency_seconds < 0.0
        ):
            raise ValueError("mean_latency_seconds must be finite and not negative")

        self._validate_token_count(
            name="total_input_tokens",
            value=self.total_input_tokens,
        )
        self._validate_token_count(
            name="total_output_tokens",
            value=self.total_output_tokens,
        )

    @staticmethod
    def _validate_score(
        name: str,
        value: float,
        minimum: float,
        maximum: float,
    ) -> None:
        if (
            not math.isfinite(value)
            or not minimum <= value <=maximum
        ):
            raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")

    @staticmethod
    def _validate_token_count(
        name: str,
        value: int | None,
    ) -> None:
        if value is not None and value < 0:
            raise ValueError(f"{name} must not be negative")